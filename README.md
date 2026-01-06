# Radioclub-Bot
running on amvera.ru

telegram bot on aiogram 3.4.1 using sentence embedding (sentence-transformers/LaBSE) for finding most similar sentence by vector and VGG-19 for image style transfer

## Functions:

* tag bot with @radioclub_bot and it will reply with random quote from Radiohead song
* send photo with caption @radioclub_bot and bot will change style of the sended photo to one of the 9 Radiohead albums' covers
* send command /message and bot will save message you send to special file for administrators
* send command /similar and bot will find Radiohead lyric with vector similar to that one of the sended text

## Project structure:

* main.py — main file with bot code, initializing test.py and similar_line.py
* style_transfer.py — file with code for style trasferring
* similar_line.py — file with code reading vectors from df, function find_similar  finds vector of the user's message and then get the line with most similar vector
* lyrics_with_vectors.csv — dataset of lyrics with vectors
* directory covers/ — dataset of nine photos for style transferring
* requirements.txt — modules and packages needed for correct work
