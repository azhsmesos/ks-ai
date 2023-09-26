import openai

openai.api_key = "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"

def tuning():
    file_response = openai.File.create(
        file=open("mydata.jsonl", "rb"),
        purpose='fine-tune'
    )
    file_id = file_response['id']
    openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")


def list():
    l = openai.FineTuningJob.list(limit=10)
    print("微调list: ", l)

def event_list():
    l = openai.FineTuningJob.list_events(id="chatcmpl-7xttQSPOkWfZ3foz9ymfTg9ZgzKKx", limit=10)

def tune_no_flie():
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": ""},
            {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}
        ]
    )
    print(completion.choices[0].message)

if __name__ == '__main__':
    tuning()
    # list()
    # event_list()
    # tune_no_flie()


