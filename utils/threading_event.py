# Необходимо для восстановления позиции курсора в консоли и приглашения ввести вопрос.
# Иначе фоновые сообщения парсера сбивают позиции ввода и курсор

import sys
import threading

from utils.current_time import get_current_time

input_active = threading.Event()


def reprint_prompt():
    if input_active.is_set():
        sys.stdout.write(f"\n[{get_current_time()}] Чего изволите, милорд?!: ")
        sys.stdout.flush()
