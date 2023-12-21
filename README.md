# TODO

-[x] Успешно запустить обучение на example датасете 

-[ ] Придумать задачу под ControlNet с существующим датасетом

-[ ] Обучить контролнет, попробовать различные техники



# Советы

1) добавляем ControlNet к существующему чекпоинту stable diffusion
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
2) потом можем запускать обучение tutorial_train
3) чтобы прочекать датасет - tutorial_dataset


AnyDoor изучить paper