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
-[ ] Порядка 3M изображений для обучения Canny Controlnet нужно было
-[ ] Есть две версии controlnet
- Note that 3k to 7k steps is not very large, and you should consider larger batch size rather than more training steps.
- If you can observe the "sudden converge" at 3k step using batch size 4,
- then, rather than train it with 300k further steps, a better idea is to use 100× gradient accumulation to 
- re-train that 3k steps with 100× batch size. Note that perhaps we should not do this
- too extremely (perhaps 100x accumulation is too extreme), but you should consider that,
- since "sudden converge" will always happen at that certain point, getting a better converge is more important.
- But usually, if your logic batch size is already bigger than 256, then further extending the batch size is not very meaningful.
- In that case, perhaps a better idea is to train more steps

# Советы

0) conda activate control
1) добавляем ControlNet к существующему чекпоинту stable diffusion
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
2) потом можем запускать обучение tutorial_train
3) чтобы прочекать датасет - tutorial_dataset


AnyDoor изучить paper