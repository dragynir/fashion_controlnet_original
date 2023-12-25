# TODO

-[x] Успешно запустить обучение на example датасете 

-[x] Придумать задачу под ControlNet с существующим датасетом (iMaterialist(Fashion))

-[x] Делаю EDA датасета, чтобы понимать что в нем лежит

-[x] Пишу код для энкодинга масок (по частям тела(голова,верх,низ)), как тут (https://github.com/levindabhi/cloth-segmentation/tree/main)

-[x] Извлекаю из датасета промпты с названием одежды (будем их подавать на вход при обучении)

-[ ] Ставлю clip-interrogate на винду

https://www.kaggle.com/discussions/general/74235 - kaggle датасет в colab
-[ ] Гуглить: extarct prompt from image

Мой colab с clip: https://colab.research.google.com/drive/1fbqojIlYDpf9HYwPA3Noe4c4ikeW-EHM

Прогоняю CLIP или аналогичную модель чтобы получить промпты, также можно Grounding dino или
1) CLIP (https://github.com/pharmapsychotic/clip-interrogator)
   2) Сначала пробую fast mode
   3) Потом создаю промпты с помощью best
2) Kosmos2 https://github.com/microsoft/unilm/tree/master/kosmos-2

 (https://huggingface.co/spaces/ydshieh/Kosmos-2)

-[ ] Замешиваю промпты из датасета с новыми промптами

-[ ] Обучаю контролнет на масках сегментации (пробую различные техники)

- [ ] Попробовать различные составления промптов(можно одежду в спец символы выделять)

- [ ] Тестируюсь на тестовом датасете (может там есть маски уже)

-[ ] Далее обучаю unet, используя этот репозиторий - чтобы можно было юзать на новых изображениях (https://github.com/levindabhi/cloth-segmentation/tree/main)
-[ ] Можно попробовать ее сначала найти - может есть где-то все же 

-[ ] Пишу интерфейс с gradio или streamlit

-[ ] Пробую добавить openpose


# Замечания

-[ ] В некоторых изображениях размечены не все люди
-[ ] Обучение на 1 карточке 3090 будет примерно 5 дней
-[ ] Есть две версии controlnet


# Советы

0) conda activate control
1) добавляем ControlNet к существующему чекпоинту stable diffusion
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
2) потом можем запускать обучение tutorial_train
3) чтобы прочекать датасет - tutorial_dataset


AnyDoor изучить paper