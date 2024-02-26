import openai
import config

openai.api_key = config.authorization


# Completion
def get_completion(prompt, model="gpt-4-1106-preview", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content


# Prompt
def prompt_and_complete(summary_length, text_to_summarise):
    prompt = f"""
    Your task is to generate a short summary of the text below (delimited by triple backticks) in at most {summary_length} words. 

    The text: ```{text_to_summarise}```
    """

    return get_completion(prompt)


# Batch a string into a correctly sized list of words
def batch_list(text, max_batch_size):
    word_list = text.split()
    word_batches = []
    word_batch = ""
    for word in word_list:
        if len((word_batch + word).split()) <= max_batch_size:
            word_batch += word + " "
        else:
            word_batches.append(word_batch)
            word_batch = word + " "
    word_batches.append(word_batch)

    return word_batches


def batch_summariser(text, batch_summary_length, max_batch_size):
    word_batches = batch_list(text, max_batch_size)

    # Loop through the list of batches and summarise each of them
    batch_summaries = []
    for word_batch in word_batches:
        success = False
        timeout_counter = 0
        while not success:
            if timeout_counter < 10:
                try:
                    response = prompt_and_complete(batch_summary_length, word_batch)
                    batch_summaries.append(response)
                    success = True
                except:
                    pass
            else:
                break

    # Create a single string of all the summaries
    batch_summary = " ".join(batch_summaries)

    return batch_summary


def summarise(text_to_summarise, summary_length):
    # Settings
    max_batch_size = 600  # Number of words that are equivalent to the max number of tokens CHAT-GPT allows
    small_batch_initial_summary_length = 30  # Summary length produced for each batch

    # Read the text and split it into a list of words
    text = str(text_to_summarise.replace("\n", ""))

    batch_summary = text  # The summarised version is intially the whole text until we measure its length and discover it is too long
    intermediate_passes = 0
    for i in range(1, 10):  # Maximum number of summaries of summaries of summaries
        if i == 1:
            print(
                f"In large loop {i} the number of words in the batch summary is {len(batch_summary.split())}"
            )

        current_batch_size = len(
            " ".join(batch_list(batch_summary, max_batch_size)).split()
        )

        if current_batch_size > max_batch_size:
            batch_summary = batch_summariser(
                batch_summary, small_batch_initial_summary_length * i, max_batch_size
            )  # Make the summary length longer when there are fewer of them to go on
            intermediate_passes += 1
            print(
                f"After intermediate pass {intermediate_passes} the number of words in the batch summary is {len(batch_summary.split())}"
            )

        else:  # Final summarisation and file output
            batch_summary = batch_summariser(
                batch_summary, summary_length, max_batch_size
            )
            print(
                f"The number of intermediate summarisation passes was: {intermediate_passes}"
            )
            break

    return batch_summary
