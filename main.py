import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin, urlparse, parse_qs
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE = "https://aplikacje.nfz.gov.pl"

# Sesja z connection pooling
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=3
)
session.mount('http://', adapter)
session.mount('https://', adapter)
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
})

# Cache dla powtarzających się requestów
@lru_cache(maxsize=1000)
def fetch_cached(url, referer=None):
    """Wersja fetch z cachowaniem identycznych URL"""
    return fetch(url, referer)

def fetch(url, referer=None, max_retries=3, timeout=10):
    """Zoptymalizowany fetch z krótszym timeoutem"""
    headers = {"Referer": referer} if referer else {}
    
    for attempt in range(max_retries):
        try:
            resp = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                return None
            wait = 0.5 * (2 ** attempt)  # Szybszy backoff
            time.sleep(wait)
    return None

def get_first_table_soup(html):
    """Zoptymalizowane parsowanie - tylko pierwsza tabela"""
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")  # lxml jest szybszy niż html.parser
    return soup.find("table")

def html_table_to_df_and_links(table):
    """Szybsza konwersja tabeli do DataFrame"""
    if table is None:
        return pd.DataFrame(), []

    # Szybsze parsowanie nagłówków
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    
    rows = []
    links = []
    
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
            
        row_vals = [td.get_text(strip=True).replace('\r', ' ').replace('\n', ' ') for td in tds]
        rows.append(row_vals)
        
        # Link tylko z pierwszej kolumny
        a = tds[0].find("a") if tds else None
        first_link = urljoin(BASE, a.get("href")) if a and a.get("href") else None
        links.append(first_link)

    if not rows:
        return pd.DataFrame(), []
    
    df = pd.DataFrame(rows, columns=headers if headers and len(headers) == len(rows[0]) else None)
    return df, links

def get_all_providers_parallel(year=2024, branch="06", service="03", max_workers=5):
    """Równoległe pobieranie stron świadczeniodawców"""
    
    def fetch_page(page_num):
        url = f"{BASE}/umowy/Provider/SearchResults?Year={year}&Branch={branch}&ServiceType={service}&page={page_num}"
        html = fetch(url, referer=f"{BASE}/umowy/Provider/Search")
        table = get_first_table_soup(html)
        if table:
            df_page, _ = html_table_to_df_and_links(table)
            return df_page if not df_page.empty else None
        return None
    
    # Sprawdź ile jest stron (można też zacząć od 1 i pobierać aż będzie puste)
    providers = []
    
    # Najpierw sprawdź pierwszą stronę żeby poznać zakres
    first_page = fetch_page(1)
    if first_page is not None:
        providers.append(first_page)
        
        # Próbuj pobierać kolejne strony równolegle
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_page, page): page for page in range(2, 100)}  # max 100 stron
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Pobieranie świadczeniodawców"):
                result = future.result()
                if result is not None:
                    providers.append(result)
                else:
                    break  # Jeśli trafimy na pustą stronę, przerywamy
    
    if not providers:
        return pd.DataFrame()
    
    df_all = pd.concat(providers, ignore_index=True)
    
    # Zachowaj tylko istotne kolumny
    cols_to_keep = ["Kod", "Nazwa świadczeniodawcy", "Miasto"]
    available_cols = [c for c in cols_to_keep if c in df_all.columns]
    
    if not available_cols:
        available_cols = df_all.columns[:3].tolist()
    
    return df_all[available_cols].copy()

def process_single_provider(prov_row, year, branch, service):
    """Przetwarzanie pojedynczego świadczeniodawcy - do równoległego wykonania"""
    prov_code_raw = prov_row.get("Kod", "").strip()
    if not prov_code_raw:
        return []
    
    prov_code_encoded = prov_code_raw.replace("/", "%2F")
    prov_name = prov_row.get("Nazwa świadczeniodawcy", "")
    prov_city = prov_row.get("Miasto", "")
    
    records = []
    
    try:
        # Pobierz umowy
        url = f"{BASE}/umowy/Agreements/GetAgreements?Year={year}&ServiceType={service}&Code={prov_code_encoded}&Branch={branch}"
        referer = f"{BASE}/umowy/Provider/Details?Code={prov_code_raw}&Year={year}&Branch={branch}&ServiceType={service}"
        html = fetch(url, referer=referer)
        
        table = get_first_table_soup(html)
        agreements_df, links = html_table_to_df_and_links(table)
        
        if agreements_df.empty:
            return []
        
        agreements_df["__details_link"] = links
        
        # Przetwórz umowy
        for _, agr in agreements_df.iterrows():
            details_link = agr.get("__details_link")
            if not details_link:
                continue
            
            # Pobierz plany
            html = fetch(details_link, referer=f"{BASE}/umowy/Agreements/Details")
            table = get_first_table_soup(html)
            df_plans, plan_links = html_table_to_df_and_links(table)
            
            if df_plans.empty:
                continue
            
            # Znajdź kolumnę z kodem produktu
            product_col = next((c for c in df_plans.columns if "produktu" in c.lower() and "kod" in c.lower()), None)
            
            for i, plan_row in df_plans.iterrows():
                product_link = plan_links[i] if i < len(plan_links) else None
                product_code = plan_row.get(product_col) if product_col else None
                
                if not product_link:
                    continue
                
                # Pobierz miesiące
                html = fetch(product_link, referer=f"{BASE}/umowy/AgreementPlans/Details?Code={prov_code_raw}")
                table = get_first_table_soup(html)
                months_df, _ = html_table_to_df_and_links(table)
                
                if months_df.empty:
                    continue
                
                # Znajdź kod umowy
                kod_umowy = None
                for k, v in agr.items():
                    if "umowy" in k.lower() and "kod" in k.lower():
                        kod_umowy = v
                        break
                
                # Twórz rekordy
                for _, mrow in months_df.iterrows():
                    rec = {
                        "Rok": year,
                        "Kod świadczeniodawcy": prov_code_raw,
                        "Nazwa świadczeniodawcy": prov_name,
                        "Miasto": prov_city,
                        "Kod umowy": kod_umowy,
                        "Kod produktu kontraktowanego": product_code,
                    }
                    rec.update(mrow.to_dict())
                    records.append(rec)
                    
    except Exception as e:
        logger.error(f"Błąd dla świadczeniodawcy {prov_code_raw}: {e}")
    
    return records

def run_pipeline_parallel(year=2024, branch="06", service="03", max_workers=10, save_as="NFZ_full_{}.xlsx"):
    """Główny pipeline z równoległym przetwarzaniem"""
    
    # 1. Pobierz listę świadczeniodawców (równolegle)
    logger.info("Pobieranie listy świadczeniodawców...")
    providers_df = get_all_providers_parallel(year, branch, service, max_workers=5)
    
    if providers_df.empty:
        logger.warning("Brak świadczeniodawców")
        return
    
    logger.info(f"Znaleziono {len(providers_df)} świadczeniodawców")
    
    # 2. Przetwarzaj równolegle
    all_records = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_provider, row, year, branch, service): idx 
            for idx, row in providers_df.iterrows()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Przetwarzanie"):
            records = future.result()
            if records:
                all_records.extend(records)
    
    if not all_records:
        logger.warning("Brak zebranych danych")
        return
    
    # 3. Twórz DataFrame i zapisz
    df_final = pd.DataFrame(all_records)
    
    # Czyszczenie danych
    if 'Kod umowy' in df_final.columns:
        df_final['Kod umowy'] = df_final['Kod umowy'].str.split(' ').str[0]
    
    if 'Kod produktu kontraktowanego' in df_final.columns:
        df_final['Kod produktu kontraktowanego'] = df_final['Kod produktu kontraktowanego'].str.split(' ').str[0]
    
    filename = save_as.format(year)
    df_final.to_excel(filename, index=False)
    logger.info(f"Zapisano {len(df_final)} wierszy do: {filename}")
    
    return df_final

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NFZ Data Scraper')
    parser.add_argument('--year', type=int, default=2024, help='Rok do pobrania')
    parser.add_argument('--branch', type=str, default="06", help='Oddział NFZ')
    parser.add_argument('--service', type=str, default="03", help='Typ świadczenia')
    parser.add_argument('--workers', type=int, default=10, help='Liczba równoległych wątków')
    parser.add_argument('--output', type=str, default="NFZ_full_{}.xlsx", help='Nazwa pliku wyjściowego')
    
    args = parser.parse_args()
    
    logger.info(f"Rozpoczynam scraping: rok={args.year}, oddział={args.branch}, typ={args.service}")
    
    df = run_pipeline_parallel(
        year=args.year, 
        branch=args.branch, 
        service=args.service, 
        max_workers=args.workers,
        save_as=args.output
    )
    
    if df is not None:
        logger.info(f"Zakończono pomyślnie. Zebrano {len(df)} rekordów.")
    else:
        logger.error("Scraping zakończony błędem.")
