import torch
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from transformers import pipeline

# Define a pipeline for inference using an LLM
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# Test the pipeline manually
outputs = pipe("Tell me something about wallabies", max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])

# Start Command
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Hi! I am your AI assistant. How can I help you today?")

# Process Message
async def process(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text
    response = pipe(user_input, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    await update.message.reply_text(response[0]["generated_text"])

# Main Function
def main():
    API_TOKEN = "7791816134:AAGuq2SGJ7Tdfb4OPeWW_ndgmUftMGIwBZQ"  # my personal token
    application = Application.builder().token(API_TOKEN).build()

    # Add Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process))

    # Run the Bot
    application.run_polling()

if __name__ == "__main__":
    main()


