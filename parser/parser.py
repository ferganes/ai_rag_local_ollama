from typing import List, Dict, Any
import time

import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import database_manager.database_manager as database_manager
import utils.current_time as current_time
from config import URL_FEED
from utils.threading_event import reprint_prompt


def extract_rss_feed(url: str) -> str | None:
    """
        Возвращает RSS фид.

        Args:
            url (str): Адрес URL RSS фида

        Returns:
            str: xml или None

    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        if response.text:
            print("\n--> RSS фид успешно получен...")
            return response.text

    except requests.exceptions.RequestException:
        return None


def transform_rss_to_list_dict(xml_string: str) -> List[dict] | None:
    """
    Преобразует XML RSS в список словарей.

    Args:
        xml_string (str): Строка с XML данными RSS.

    Returns:
        list: Список словарей

    """
    try:
        root = ET.fromstring(xml_string)

        items = root.findall('.//item')
        result = []

        for item in items:
            title_elem = item.find('title')
            link_elem = item.find('link')
            pub_date_elem = item.find('pubDate')
            enclosure_elem = item.find('enclosure')

            item_data = {
                'title': title_elem.text if title_elem is not None else None,
                'link': link_elem.text if link_elem is not None else None,
                'pub_date': pub_date_elem.text if pub_date_elem is not None else None,
                'image_url': enclosure_elem.get('url') if enclosure_elem is not None else None,
                'full_text': ''
            }

            result.append(item_data)

        return result

    except ET.ParseError:
        return None


def extract_article(url: str) -> str | None:
    """
    Парсит текст из селекторов 'topic-body__content-text' по переданному url.
    Объединяет куски текста в один и возвращает как строку

    Args:
        url (str)

    Returns:
        str или None
    """
    try:

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        }

        # Отправляем GET-запрос
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Парсим текст новости
        soup = BeautifulSoup(response.text, 'html.parser')

        # Находим все элементы с классом 'topic-body__content-text'
        content_elements = soup.find_all(class_='topic-body__content-text')

        # Извлекаем текст каждого элемента
        full_text = []

        for element in content_elements:
            element_text = element.get_text().strip()

            if element_text:
                full_text.append(element_text)

        return ' '.join(full_text)

    except Exception:
        return None


def get_existing_links(db: Chroma) -> set[str]:
    """
    Функция возвращает set уникальных ссылок из метаданных имеющихся в базе документов

    Args: db: Chroma

    Return: set(str)
    """

    existing_docs = db.get()
    existing_links = set()

    if existing_docs and 'metadatas' in existing_docs:
        for metadata in existing_docs['metadatas']:
            if metadata and 'link' in metadata:
                existing_links.add(metadata['link'])

    return existing_links


def split_text_to_docs(items: List[Dict[str, Any]]) -> List[Document]:
    """
    Разбивает текст на чанки Document

    Args: Dict[str, Any]

    Returns: List[Document]
    """

    texts, metadatas = [], []
    for item in items:
        text = item.get('full_text', '').strip()
        if not text:
            continue

        texts.append(text)
        metadatas.append({
            'source': item.get('link', 'unknown'),
            'title': item.get('title', ''),
            'link': item.get('link', ''),
            **{k: v for k, v in item.items() if k != 'full_text'}
        })

    if not texts:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    return text_splitter.create_documents(texts=texts, metadatas=metadatas)


def parsing_worker(db, interval=900):

    while True:
        print(f"\n[{current_time.get_current_time()}] Запускаю парсинг в фоне...")

        rss_feed = extract_rss_feed(URL_FEED)

        parsed_items = transform_rss_to_list_dict(rss_feed)

        existing_links = get_existing_links(db)

        # Фильтруем по ключу link, убираем дубликаты
        filtered_parsed_items = [
            item for item in parsed_items
            if item.get('link') not in existing_links
        ]

        print(f'\n--> Документов в базе {len(existing_links)}, статей в RSS {len(parsed_items)}, '
              f'новых статей {len(filtered_parsed_items)}')

        if filtered_parsed_items:

            for item in filtered_parsed_items:
                item['full_text'] = extract_article(item['link'])
                print(f'{"OK | Получен" if item["full_text"] else "X Не получилось получить"} '
                      f'полный текст | url: {item["link"]}')

            docs = split_text_to_docs(filtered_parsed_items)

            print(f"\n[{current_time.get_current_time()}] Обновление базы...")
            database_manager.update_database(db, docs)

            print(f"\n[{current_time.get_current_time()}] База обновлена...")
        else:
            print(f"\n[{current_time.get_current_time()}] Новых статей нет...")

        reprint_prompt()
        time.sleep(interval)
