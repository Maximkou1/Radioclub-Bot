import asyncio
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram import F
from aiogram.types import Message

from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

import os
import random

from similar_line import *


# достаём токен из .env-файла
load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')
if not API_TOKEN:
    exit("Error: API_TOKEN not found in .env file")

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


# инициализируем модель
model = SentenceTransformer('sentence-transformers/LaBSE')

# записываем все строчки из датасета в переменную sentences
df = pd.read_csv('lyrics_with_vectors.csv')
sentences = df['Lyrics'].tolist()

# достаём вектора всех строчек из датасета
embeddings = df['Vectors'].apply(eval).apply(np.array)
# и делаем из них читаемый массив
embed_list = []
for i in range(len(embeddings)):
    embed_list.append(embeddings[i])
embeddings = np.array(embed_list)


HELP_COMMAND = """
Hi! I'm Radioclub bot
• tag me and I'll answer with Radiohead quote
• tag me and send any photo — I'll change its style to Radiohead's album cover
• write /message to ask or write something to admins
• write /similar and then some text — I'll respond with similar Radiohead lyric
"""


# функция для запуска файла
def execute_python_file(file_path):
    try:
        os.system(f'python {file_path}')
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")


class Form(StatesGroup):
    text = State()


class Form2(StatesGroup):
    text2 = State()


# два хэндлера (finite state machine) для записи сообщений администраторам
@dp.message(F.text, Command("message"))
async def cmd_message(message: Message, state: FSMContext):
    await message.answer("Please write your message to administrators")
    await state.set_state(Form.text)


@dp.message(Form.text)
async def get_message(message: Message,  state: FSMContext):
    with open('messages.txt', 'a') as fw:
        fw.write(str(message.from_user.id)+' '+message.text+'\n')
    await state.update_data(text=message.text)
    await message.reply("Thank you. Your message is saved")
    await state.clear()


# два хэндлера для поиска строк с векторами, похожими на сообщение
@dp.message(F.text, Command("similar"))
async def cmd_message(message: Message, state: FSMContext):
    await message.answer("Write anything and I respond with similar lyrics")
    await state.set_state(Form2.text2)


@dp.message(Form2.text2)
async def get_message(message: Message,  state: FSMContext):
    text = message.text
    await state.update_data(text2=message.text)
    await message.reply(find_similars(text))
    await state.clear()


# хэндлер для команды старт
@dp.message(F.text, Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("hemlo my name is tom yormk\nsend /help for more info")


# хэндлер для команды помощь
@dp.message(F.text, Command("help"))
async def cmd_help(message: Message):
    await message.answer(HELP_COMMAND)


# хэндлер для вывода случайной строчки
@dp.message(F.text)
async def lyrics(message: Message):
    if "@radioclub_bot" in message.text:
        line = random.choice(sentences)
        await message.answer(line)


# хэндлер для обработки фото
@dp.message(F.photo, F.caption == "@radioclub_bot")
async def get_photo(message: types.Message):
    await message.bot.download(file=message.photo[-1].file_id, destination='images/test.jpg')
    await message.reply("got it. please wait")
    execute_python_file("./style_transfer.py")
    await message.answer_photo(types.FSInputFile(path="images/result.jpg"), caption="Here!")


# запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
