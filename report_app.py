import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from PIL import Image

# Pricing dictionary (you can edit or expand this)
PRICING = {
  "gpt-4.1": {
    "prompt":     { "price_usd": 2.00,  "multiplier": 1000000 },
    "completion": { "price_usd": 8.00,  "multiplier": 1000000 }
  },

  "gpt-4.1-mini": {
    "prompt":     { "price_usd": 0.40,  "multiplier": 1000000 },
    "completion": { "price_usd": 1.60,  "multiplier": 1000000 }
  },

  "gpt-4.1-nano": {
    "prompt":     { "price_usd": 0.10,  "multiplier": 1000000 },
    "completion": { "price_usd": 0.40,  "multiplier": 1000000 }
  },

  "gpt-4.5-preview": {
    "prompt":     { "price_usd": 75.00, "multiplier": 1000000 },
    "completion": { "price_usd":150.00, "multiplier": 1000000 }
  },

  "gpt-4o": {
    "prompt":     { "price_usd": 2.50,  "multiplier": 1000000 },
    "completion": { "price_usd":10.00,  "multiplier": 1000000 }
  },

  "gpt-4o-realtime-preview-2024-12-17": {
    "prompt":     { "price_usd": 5.00,  "multiplier": 1000000 },
    "completion": { "price_usd":20.00,  "multiplier": 1000000 }
  },

  "gpt-4o-mini": {
    "prompt":     { "price_usd": 0.15,  "multiplier": 1000000 },
    "completion": { "price_usd": 0.60,  "multiplier": 1000000 }
  },

  "gpt-4o-mini-realtime-preview-2024-12-17": {
    "prompt":     { "price_usd": 0.60,  "multiplier": 1000000 },
    "completion": { "price_usd": 2.40,  "multiplier": 1000000 }
  },

  "o1-2024-12-17": {
    "prompt":     { "price_usd":15.00,  "multiplier": 1000000 },
    "completion": { "price_usd":60.00,  "multiplier": 1000000 }
  },

  "o1-pro-2025-03-19": {
    "prompt":     { "price_usd":150.00, "multiplier": 1000000 },
    "completion": { "price_usd":600.00, "multiplier": 1000000 }
  },

  "o3-2025-04-16": {
    "prompt":     { "price_usd":10.00,  "multiplier": 1000000 },
    "completion": { "price_usd":40.00,  "multiplier": 1000000 }
  },

  "o4-mini": {
    "prompt":     { "price_usd": 1.10,  "multiplier": 1000000 },
    "completion": { "price_usd": 4.40,  "multiplier": 1000000 }
  },

  "o3-mini-2025-01-31": {
    "prompt":     { "price_usd": 1.10,  "multiplier": 1000000 },
    "completion": { "price_usd": 4.40,  "multiplier": 1000000 }
  },

  "o1-mini-2024-09-12": {
    "prompt":     { "price_usd": 1.10,  "multiplier": 1000000 },
    "completion": { "price_usd": 4.40,  "multiplier": 1000000 }
  },

  "codex-mini-latest": {
    "prompt":     { "price_usd": 1.50,  "multiplier": 1000000 },
    "completion": { "price_usd": 6.00,  "multiplier": 1000000 }
  },

  "gpt-4o-mini-search-preview-2025-03-11": {
    "prompt":     { "price_usd": 0.15,  "multiplier": 1000000 },
    "completion": { "price_usd": 0.60,  "multiplier": 1000000 }
  },

  "gpt-4o-search-preview-2025-03-11": {
    "prompt":     { "price_usd": 2.50,  "multiplier": 1000000 },
    "completion": { "price_usd":10.00,  "multiplier": 1000000 }
  },

  "computer-use-preview-2025-03-11": {
    "prompt":     { "price_usd": 3.00,  "multiplier": 1000000 },
    "completion": { "price_usd":12.00,  "multiplier": 1000000 }
  },
  "o3": {
    "prompt":     { "price_usd": 2.00,  "multiplier": 1000000 },
    "completion": { "price_usd": 8.00,  "multiplier": 1000000 }
  }
}



def per_token(block): return block["price_usd"] / block["multiplier"]

# --- Report Generator ---
def generate_report(df, experiment_name, model_name, subject, prompt_cost, completion_cost):
    filtered_df = df[df["experiment_name"] == experiment_name].copy()
    if filtered_df.empty:
        st.warning("⚠️ No data found for the selected experiment.")
        return

    # Token and cost calculations
    filtered_df["prompt_tokens"] = pd.to_numeric(filtered_df["prompt_tokens"], errors="coerce")
    filtered_df["completion_tokens"] = pd.to_numeric(filtered_df["completion_tokens"], errors="coerce")
    filtered_df["row_cost_usd"] = filtered_df["prompt_tokens"] * prompt_cost + filtered_df["completion_tokens"] * completion_cost
    total_prompt = int(filtered_df["prompt_tokens"].sum())
    total_completion = int(filtered_df["completion_tokens"].sum())
    total_cost = round(filtered_df["row_cost_usd"].sum(), 4)

    # Latency
    filtered_df["latency"] = pd.to_numeric(filtered_df["latency"], errors="coerce")
    avg_latency = round(filtered_df["latency"].mean(), 2)
    min_latency = round(filtered_df["latency"].min(), 2)
    max_latency = round(filtered_df["latency"].max(), 2)

    # Consistency
    filtered_df["student_response_id"] = filtered_df.apply(
        lambda row: f"{row['experiment_name']}_{row['Assessment Name']}_{row['Student Name']}", axis=1
    )
    grouped = filtered_df.groupby("student_response_id")["AI Grade"].nunique()
    consistent = (grouped == 1).sum()
    total = len(grouped)
    consistency_pct = round((consistent / total) * 100, 2)

    # Agreement categories
    agreements = filtered_df["category"].value_counts()
    aligned_pct = round((agreements.get("aligned", 0) / len(filtered_df)) * 100, 2)
    lenient_pct = round((agreements.get("lenient", 0) / len(filtered_df)) * 100, 2)
    strict_pct = round((agreements.get("strict", 0) / len(filtered_df)) * 100, 2)

    # --- Plot the report ---
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    summary = f"""
    **Experiment:** {experiment_name}  
    **Model:** {model_name}  
    **Subject:** {subject}  
    **Prompt Tokens:** {total_prompt:,}  
    **Completion Tokens:** {total_completion:,}  
    **Total Cost:** ${total_cost}  
    **Consistency:** {consistency_pct}%  
    **Latency (avg/min/max):** {avg_latency}s / {min_latency}s / {max_latency}s
    """
    ax0.text(0, 1, summary, fontsize=12, va="top")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.pie([aligned_pct, lenient_pct, strict_pct], labels=["Aligned", "Lenient", "Strict"],
            autopct="%1.1f%%", colors=["green", "orange", "red"], startangle=140)
    ax1.set_title("AI-Human Agreement")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(["Consistent", "Inconsistent"], [consistency_pct, 100 - consistency_pct],
            color=["blue", "gray"])
    ax2.set_title("AI Consistency")
    ax2.set_ylabel("Percentage")

    ax3 = fig.add_subplot(gs[1, 1])
    filtered_df["latency"].dropna().hist(bins=20, ax=ax3, color="purple")
    ax3.set_title("Latency Distribution")
    ax3.set_xlabel("Latency (s)")
    ax3.set_ylabel("Frequency")

    plt.tight_layout()

    # Save to subject folder
    folder = f"reports/{subject}_reports"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{experiment_name}_final_report.png")
    plt.savefig(path)
    st.pyplot(fig)
    st.success(f"✅ Report saved at: {path}")

# --- Past Reports Viewer ---
def list_past_reports():
    st.markdown("## 🗂️ Browse Previous Reports")
    subjects = ["math", "ela"]
    cols = st.columns(len(subjects))

    for idx, subj in enumerate(subjects):
        with cols[idx]:
            st.subheader(f"📁 {subj.upper()}")
            folder = f"reports/{subj}_reports"
            if not os.path.isdir(folder):
                st.info("No reports yet.")
                continue
            files = sorted(os.listdir(folder), reverse=True)
            for file in files:
                path = os.path.join(folder, file)
                with st.container(border=True):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(path, use_column_width=True)
                    with col2:
                        st.markdown(f"**{file}**")
                        st.caption(f"{subj.upper()} Report")
                        with open(path, "rb") as f:
                            st.download_button("📥 Download", f, file_name=file, mime="image/png", key=f"{subj}_{file}")

# --- Main UI ---
def main():
    st.set_page_config(layout="wide")
    st.title("📊 AI Grading Report Generator")
    st.markdown("> Upload a CSV of graded experiments. Select an experiment and subject to instantly view a report.")

    uploaded_file = st.file_uploader("📤 Upload `graded_data.csv`", type="csv", key="uploader")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "experiment_name" not in df.columns:
            st.error("Missing required column: `experiment_name`")
            return

        experiment_names = sorted(df["experiment_name"].dropna().unique().tolist())
        subjects = ["math", "ela"]

        # Detect experiment/subject state
        if "experiment_list" not in st.session_state or st.session_state.experiment_list != experiment_names:
            st.session_state.experiment_list = experiment_names
            st.session_state.experiment = experiment_names[0]
            st.session_state.subject = subjects[0]

        def update_experiment(): st.session_state.experiment = st.session_state.selected_experiment
        def update_subject(): st.session_state.subject = st.session_state.selected_subject

        st.selectbox("🧪 Select Experiment", experiment_names, index=experiment_names.index(st.session_state.experiment),
                     key="selected_experiment", on_change=update_experiment)
        st.selectbox("📚 Select Subject", subjects, index=subjects.index(st.session_state.subject),
                     key="selected_subject", on_change=update_subject)

        selected_df = df[df["experiment_name"] == st.session_state.experiment]
        if "model" in df.columns and not selected_df["model"].dropna().empty:
            model_name = selected_df["model"].dropna().iloc[0]
            st.success(f"✅ Auto-detected model: {model_name}")
        else:
            model_name = st.text_input("Enter model name (e.g. gpt-4o-2024-08-06)")

        if model_name not in PRICING:
            st.warning(f"Unknown model: {model_name}. Valid: {', '.join(PRICING)}")
            return

        # Costs
        prompt_cost = per_token(PRICING[model_name]["prompt"])
        completion_cost = per_token(PRICING[model_name]["completion"])

        st.subheader("📈 Report Visualization")
        generate_report(df, st.session_state.experiment, model_name, st.session_state.subject, prompt_cost, completion_cost)

    # Past reports viewer (always shown)
    list_past_reports()

if __name__ == "__main__":
    main()