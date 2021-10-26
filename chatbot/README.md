Over the weekend, I built an interactive chat-bot to interface with the real-time chat in a video game. This was for educational purposes. Disclaimer: it might be against the game's ToS to interface event data with a 3rd party app. The difficult part was getting real-time event data from the game app :)



## Purpose

The purpose of this application was to see if I could build a "meme-generator" based on common responses and feedback found in the historical chat log. Of course, the quality of the the output depends on the prompt that is feed into the model.



## Training

As always, the data is "dirty". In a free-form chat, there can be multiple conversations going on, bad spelling, emojis based punctuation, etc. The text generation model uses a standard Embedding+LSTM+CRF. 



For training, I concatenate each seed message with the next 6 potential responses within in a minute, and then create a target variable for each token in the response. I used a special token to separate the messages.



 There was also some data processing to remove spam messages and clean up the text\punctuation, both for historical and real-time chat data. For real-time chat bot, I use the last 3 messages in the chat stream as an input prompt to the model. 



Finally, the message is reformatted and sent to the game client application! 



## TODO

If I were to give this more time, here are some things I would consider:

1. Train a "critic" model to rank a few probabilistic (generative) outputs, based on valid "human-vetted" responses

2. Use an attention layer instead of a LSTM (although, I have only seen attention used for neural translation before)

3. Use a name-entity-resolution (NER) algorithm to figure out where to put punctuation in the output text.


