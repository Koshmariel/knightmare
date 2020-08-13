# Knightmare

I am trying to get artificial intelligence to play KnightMare. KM is upward-scrolling shoot 'em up game which initially was made for MSX. Also there is a PC DOS remake of it     on which I had been playing when I was a kid. 

Files:
1. findFilesInFolder.py - getting paths of all files with certain extension in the folder & subfolders (optionally)
2. prepare_screenshots.py - saves screenshots from the game
3. game_mode_recognizer_ann.py - training of the ANN for the game_mode_recognizer class
4. game_mode_recognizer.py - game_mode_recognizer class, recognizes in the stage in which the game is - menu, gameplay etc.
5. digit_scrapper.py - scrap images with digits for the digit_recognizer_ann.py
6. digit_recognizer_ann.py - training of the ANN for the digit_recognizer class
7. digit_recognizer.py - digit_recognizer class
8. score_lives_recognizer.py - score_lives_recognizer class, accepts screenshot, returns current score & lives
9. dqn_lifetime_memory.py - brain: choosing actions, learning etc.
10. directkeys.py - emulating key presses
11. environment_v4.py - emulating OpenAI Gym environment
12. ai_v4.py - main script

Overview:
[![](http://img.youtube.com/vi/zj6Cd77lw5Y/0.jpg)](http://www.youtube.com/watch?v=zj6Cd77lw5Y "")
