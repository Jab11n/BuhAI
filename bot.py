import discord
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

bad_words = ['if you want to filter certain words', 'put them here in this list']

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_length=100, 
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        num_return_sequences=1
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    for word in bad_words:
        if word in response.lower():
            print(response)
            response = "Sorry, my response has been filtered. Please try again?" # if the response was filtered it gets logged to the console
            break

    return response

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    if client.user.mentioned_in(message):
        prompt = message.content.replace(f"<@{client.user.id}>", "").strip()

        if prompt:
            response = generate_response(prompt)
            await message.reply(f"{response}")
        else:
            await message.reply("Please ask me something!")

client.run('TOKEN_GOES_HERE')
