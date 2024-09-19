from googleapiclient.discovery import build
import wikipediaapi
from llama_index.core import Document
import os


def search_wikipedia(query, n_articles=5):
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

        return wikipedia_links

    except Exception as e:
        print(f"Error during search: {e}")
        return []
    

def fetch_wikipedia_pages(links,vector_index):
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
                # Store the title and the summary of the page
                doc = Document(text=page.text, id_=str(page.pageid))
                doc_chunks.append(doc)
                pages_titles.append(page_title)

            else:
                print(f"Page {page_title} does not exist on Wikipedia")

        except Exception as e:
            print(f"Error fetching page {link}: {e}")

    vector_index.refresh_ref_docs(doc_chunks)
    return pages_titles