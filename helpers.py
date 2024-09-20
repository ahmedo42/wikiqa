import logging
from googleapiclient.discovery import build
import wikipediaapi
from llama_index.core import Document
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_wikipedia(query, n_articles=5):
    """
    Search Wikipedia articles using Google Custom Search API.

    Args:
        query (str): The search query.
        n_articles (int): Number of articles to retrieve. Defaults to 5.

    Returns:
        list: A list of Wikipedia article URLs.
    """
    query = query + " en.wikipedia.org"
    try:
        # Build a service object for interacting with the API
        service = build("customsearch", "v1", developerKey=os.environ['API_KEY'])

        # Execute the search query
        result = service.cse().list(
            q=query,           
            cx=os.environ['CX'], 
            num=n_articles
        ).execute()

        wikipedia_links = []
        if 'items' in result:
            for item in result['items']:
                link = item['link']
                wikipedia_links.append(link)

        logging.info(f"Found {len(wikipedia_links)} Wikipedia articles")
        return wikipedia_links

    except Exception as e:
        logging.error(f"Error during search: {e}")
        return []

def fetch_wikipedia_pages(links, vector_index):
    """
    Fetch Wikipedia pages from provided links and update the vector index.

    Args:
        links (list): List of Wikipedia article URLs.
        vector_index: The vector index to be updated with new documents.

    Returns:
        list: A list of successfully fetched page titles.
    """
    wiki_wiki = wikipediaapi.Wikipedia('wikiQA (johndoe@example.com)', 'en')
    pages_titles = []
    doc_chunks = []

    for link in links:
        try:
            # Extract the Wikipedia page title from the URL
            page_title = link.split("/wiki/")[-1]
            
            # Fetch the Wikipedia page using wikipedia-api
            page = wiki_wiki.page(page_title)

            if page.exists():
                # Create a Document object with the page text
                doc = Document(text=page.text, id_=str(page.pageid))
                doc_chunks.append(doc)
                pages_titles.append(page_title)
                logging.info(f"Successfully fetched page: {page_title}")
            else:
                logging.warning(f"Page {page_title} does not exist on Wikipedia")

        except Exception as e:
            logging.error(f"Error fetching page {link}: {e}")

    # Update the vector index with new documents
    vector_index.refresh_ref_docs(doc_chunks)
    logging.info(f"Updated vector index with {len(doc_chunks)} new documents")
    return pages_titles