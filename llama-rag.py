import replicate

input = {
    "prompt": "Tell me something interesting about Kaohsiung, Taiwan"
}

try:
    output = replicate.run(
        "meta/meta-llama-3-8b",
        input=input
    )
    print("".join(output))
except replicate.exceptions.ReplicateError as e:
    print(f"Replicate Error: {e}")