import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a planning assistant. "
                    "Return only valid JSON with this schema: "
                    '{"steps":[{"step_id":1,"action":"action_name"}]} '
                    "Use 3 to 5 short snake_case actions."
                ),
            },
            {
                "role": "user",
                "content": "Plan a two-day Boston trip for a student on a budget."
            },
        ],
    )

    text = response.choices[0].message.content

    print("\n=== MODEL OUTPUT ===\n")
    print(text)

    output_file = OUT_DIR / "test_output.txt"
    output_file.write_text(text, encoding="utf-8")

    json_file = OUT_DIR / "test_output.json"
    try:
        parsed = json.loads(text)
        json_file.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        print(f"\nSaved raw output to: {output_file}")
        print(f"Saved parsed JSON to: {json_file}")
    except Exception:
        print(f"\nSaved raw output to: {output_file}")
        print("Could not parse output as JSON directly.")


if __name__ == "__main__":
    main()