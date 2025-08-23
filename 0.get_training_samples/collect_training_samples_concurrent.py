# %% CONCURRENT HOTPOTQA TRACE COLLECTION (process-based, checkpointed JSONL)
import os, json, itertools, tempfile, shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
import dspy
from dspy.datasets import HotPotQA
from dspy.evaluate import answer_exact_match

# ====== CONFIG ======
NUM_QUESTIONS = 50
QUESTION_SAVE_INTERVAL = 10  # checkpoint every N completed examples
DATASET_SAVE_DIR = Path("hotpotqa_agent_training_data_concurrent")
CHECKPOINT_JSONL = DATASET_SAVE_DIR / "traces.jsonl"  # append-safe, atomic writes
MAX_WORKERS = max(2, os.cpu_count() // 2)  # tune for your box
MODEL_NAME = "gemini/gemini-2.5-flash"
LM_TEMPERATURE = 0.7
EVAL_SEED = 2025
TRAIN_SIZE = 3000


# ====== UTIL ======
def atomic_append_jsonl(path: Path, rows: list[dict]):
    """Append many JSON lines atomically to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # append tmp to real file atomically
        with (
            open(path, "a", encoding="utf-8") as out,
            open(tmp, "r", encoding="utf-8") as src,
        ):
            shutil.copyfileobj(src, out)
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def load_completed_count(path: Path) -> int:
    """Count existing lines to resume; robust to partial runs."""
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ====== PER-PROCESS WORK ======
def _init_worker():
    # each worker gets its own LM + env to avoid GLOBAL_HISTORY collisions
    load_dotenv(".env")
    lm = dspy.LM(MODEL_NAME, cache=False)
    dspy.configure(lm=lm, temperature=LM_TEMPERATURE)


def _get_pred_message_responses(pred):
    # import inside worker to bind to this process' GLOBAL_HISTORY
    from dspy.clients.base_lm import GLOBAL_HISTORY

    pred_steps = len(pred.trajectory) // 4
    message_responses = []
    for pred_step in range(pred_steps):
        history_step_index = len(GLOBAL_HISTORY) - (pred_steps - pred_step) - 1
        convo = GLOBAL_HISTORY[history_step_index]["messages"].copy()
        response = (
            GLOBAL_HISTORY[history_step_index]["response"].choices[0].message.content
        )
        convo.append({"role": "assistant", "content": response})
        message_responses.append(convo)
    return message_responses


def _run_one(example):
    """
    Worker task: run agent on one example and return a list of per-step rows.
    Returns [] on failure so parent can proceed.
    """
    from hotpotqa_agent import hotpotqa_agent  # import inside worker

    q = example["question"]
    ans = example["answer"]
    rows = []
    try:
        pred = hotpotqa_agent(question=q)
        tem = answer_exact_match(example, pred)  # bool
        for step, convo in enumerate(_get_pred_message_responses(pred)):
            rows.append(
                {
                    "conversations": convo,
                    "question": q,
                    "answer": ans,
                    "step": step,
                    "trace_exact_match": tem,
                }
            )
    except Exception as e:
        # keep a minimal error record for debugging (no traceback spam)
        rows.append(
            {
                "conversations": None,
                "question": q,
                "answer": ans,
                "step": -1,
                "trace_exact_match": False,
                "error": str(e)[:500],
            }
        )
    return rows


# ====== MAIN ======
if __name__ == "__main__":
    # build/fast‑forward dataset iterator
    hotpot = HotPotQA(train_size=TRAIN_SIZE, eval_seed=EVAL_SEED)
    train_iter = iter(hotpot.train)

    # figure out resume point (how many *examples* already processed)
    # since each example can produce multiple rows, we persist an index file to resume cleanly
    # lightweight approach: also keep a sidecar counter file
    index_file = DATASET_SAVE_DIR / "examples_done.txt"
    if index_file.exists():
        start_n = int(index_file.read_text().strip())
    else:
        start_n = 0

    # fast‑forward iterator
    for _ in range(start_n):
        next(train_iter, None)

    # collect the next NUM_QUESTIONS-start_n examples up front (question+answer only)
    pending_examples = []
    for i, ex in enumerate(
        itertools.islice(train_iter, NUM_QUESTIONS - start_n), start=start_n
    ):
        pending_examples.append(
            {"idx": i, "question": ex["question"], "answer": ex["answer"]}
        )

    if not pending_examples:
        print("Nothing to do. All questions already processed.")
        raise SystemExit

    print(
        f"Starting at example #{start_n}, scheduling {len(pending_examples)} runs "
        f"with max_workers={MAX_WORKERS}."
    )

    completed_since_ckpt = 0
    buffered_rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker) as pool:
        future_map = {
            pool.submit(
                _run_one, {"question": ex["question"], "answer": ex["answer"]}
            ): ex["idx"]
            for ex in pending_examples
        }

        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                rows = fut.result()
            except Exception as e:
                # catastrophic worker failure: still make a placeholder row
                rows = [
                    {
                        "conversations": None,
                        "question": None,
                        "answer": None,
                        "step": -1,
                        "trace_exact_match": False,
                        "error": f"worker exception at idx {idx}: {e}",
                    }
                ]

            # tag rows with example index for traceability
            for r in rows:
                r["example_index"] = idx
            buffered_rows.extend(rows)

            completed_since_ckpt += 1
            # checkpoint every QUESTION_SAVE_INTERVAL examples completed
            if completed_since_ckpt >= QUESTION_SAVE_INTERVAL:
                atomic_append_jsonl(CHECKPOINT_JSONL, buffered_rows)
                # persist example count
                index_file.parent.mkdir(parents=True, exist_ok=True)
                index_file.write_text(str(idx + 1))
                print(
                    f"[ckpt] appended {len(buffered_rows)} rows at example #{idx + 1} -> {CHECKPOINT_JSONL}"
                )
                buffered_rows.clear()
                completed_since_ckpt = 0

    # final flush
    if buffered_rows:
        atomic_append_jsonl(CHECKPOINT_JSONL, buffered_rows)
        last_idx = pending_examples[-1]["idx"]
        index_file.write_text(str(last_idx + 1))
        print(
            f"[final] appended {len(buffered_rows)} rows at example #{last_idx + 1} -> {CHECKPOINT_JSONL}"
        )

    print("Done.")

    # OPTIONAL: Convert JSONL to HF Dataset when you’re ready
    # from datasets import Dataset
    # rows = [json.loads(l) for l in open(CHECKPOINT_JSONL, "r", encoding="utf-8")]
    # training_dataset = Dataset.from_list(rows)
    # training_dataset.push_to_hub(
    #     "rshn-krn/hotpotqa-agent-training-data-2",
    #     private=False,
    #     commit_message="Gemini HotPotQA agent training trace conversations (concurrent)",
    # )
