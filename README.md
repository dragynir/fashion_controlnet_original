# TODO

-[x] Успешно запустить обучение на example датасете 

-[x] Придумать задачу под ControlNet с существующим датасетом (iMaterialist(Fashion))

-[ ] Делаю EDA датасета, чтобы понимать что в нем лежит

-[ ] Пишу код для энкодинга масок (по частям тела(голова,верх,низ)), как тут (https://github.com/levindabhi/cloth-segmentation/tree/main)

-[ ] Извлекаю из датасета промпты с названием одежды (будем их подавать на вход при обучении)

-[ ] Прогоняю CLIP или аналогичную модель чтобы получить промпты

-[ ] Замешиваю промпты из датасета с новыми промптами

-[ ] Обучаю контролнет на масках сегментации (пробую различные техники)

-[ ] В качестве быстрого извлечения беру уже обученную модель ((https://github.com/levindabhi/cloth-segmentation/tree/main))

-[ ] Пишу интерфейс с gradio или streamlit



# Советы

1) добавляем ControlNet к существующему чекпоинту stable diffusion
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
2) потом можем запускать обучение tutorial_train
3) чтобы прочекать датасет - tutorial_dataset


AnyDoor изучить paper