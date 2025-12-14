import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from .agent import create_rag_agent


def run_evaluation(
    csv_path="data/eval_examples.csv",
    output_path="data/results.csv"
):
    # Load evaluation data
    df = pd.read_csv(csv_path)

    agent = create_rag_agent()
    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    predictions = []

    for _, row in df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]
        category = row["category"]

        # Run Agentic RAG
        response = agent.invoke({"messages": [{"role": "user", "content": question}]})
        answer = response["answer"]
        predictions.append(answer)

        if category == "Subjective Opinions":
            # For subjective questions, set scores to 0 as BLEU/ROUGE are not applicable
            bleu = 0.0
            rouge1 = 0.0
            rouge2 = 0.0
            rougeL = 0.0
        else:
            # For objective categories, use true_answer as reference if ground_truth is "True", else false_answer
            reference = row["true_answer"] if ground_truth == "True" else row["false_answer"]

            # BLEU
            bleu = sentence_bleu(
                [reference.split()],
                answer.split(),
                smoothing_function=smoothie
            )

            # ROUGE
            rouge_scores = rouge.score(reference, answer)
            rouge1 = rouge_scores["rouge1"].fmeasure
            rouge2 = rouge_scores["rouge2"].fmeasure
            rougeL = rouge_scores["rougeL"].fmeasure

        bleu_scores.append(bleu)
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougeL_scores.append(rougeL)

    # Save results
    df["prediction"] = predictions
    df["BLEU"] = bleu_scores
    df["ROUGE-1"] = rouge1_scores
    df["ROUGE-2"] = rouge2_scores
    df["ROUGE-L"] = rougeL_scores

    df.to_csv(output_path, index=False)
    print(f"Evaluation saved to {output_path}")

    # -----------------------------
    # PLOTS (4 CHARTS)
    # -----------------------------

    # 1) BLEU
    plt.figure()
    plt.plot(df["BLEU"])
    plt.title("BLEU Scores")
    plt.savefig("bleu.png")
    plt.clf()

    # 2) ROUGE-1
    plt.figure()
    plt.plot(df["ROUGE-1"])
    plt.title("ROUGE-1 Scores")
    plt.savefig("rouge1.png")
    plt.clf()

    # 3) ROUGE-2
    plt.figure()
    plt.plot(df["ROUGE-2"])
    plt.title("ROUGE-2 Scores")
    plt.savefig("rouge2.png")
    plt.clf()

    # 4) ROUGE-L
    plt.figure()
    plt.plot(df["ROUGE-L"])
    plt.title("ROUGE-L Scores")
    plt.savefig("rougeL.png")
    plt.clf()

    print("Charts saved: bleu.png, rouge1.png, rouge2.png, rougeL.png")


if __name__ == "__main__":
    run_evaluation()
