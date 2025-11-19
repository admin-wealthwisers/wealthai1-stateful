import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# OAuth Configuration
# ---------------------------------------------------------------------------

# Hardcoded credentials
CLIENT_ID = "1000.87PSJOW8UO5FFOT5LM7BIEL37TIO6Y"
CLIENT_SECRET = "1b154c9bd5944667a70e5eac49d399e99befd752aa"
REDIRECT_URI = "http://localhost:8000/zoho/callback"

# Zoho account credentials
ZOHO_EMAIL = "support@moneycompound.com"
ZOHO_PASSWORD = "Time@321"

# OAuth URLs
AUTH_URL = (
    "https://accounts.zoho.in/oauth/v2/auth?"
    "scope=ZohoCRM.modules.ALL,"
    "ZohoCRM.settings.ALL,"
    "ZohoCRM.coql.READ,"
    "ZohoCRM.bulk.READ,"
    "ZohoCRM.bulk.WRITE,"
    "ZohoCRM.Files.READ,"
    "ZohoCRM.org.READ,"
    "ZohoCRM.users.READ&"
    f"client_id={CLIENT_ID}&"
    "response_type=code&"
    "access_type=offline&"
    f"redirect_uri={REDIRECT_URI}&"
    "state=anyRandomCSRFToken"
)

TOKEN_URL = "https://accounts.zoho.in/oauth/v2/token"

# Global variable to store the authorization code
auth_code = None
callback_received = False

# ---------------------------------------------------------------------------
# RAG Pipeline Configuration
# ---------------------------------------------------------------------------

ARTIFACTS_ROOT = Path("artifacts/clients")
MANIFEST_PATH = Path("clients_manifest.json")

# Chunking parameters
TARGET_TOKENS = 800
OVERLAP_TOKENS = 200

# Zoho OAuth + API settings
API_DOMAIN = os.getenv("ZOHO_API_DOMAIN", "https://www.zohoapis.in")
ACCOUNTS_DOMAIN = os.getenv("ZOHO_ACCOUNTS_DOMAIN", "https://accounts.zoho.in")

# Token storage
TOKENS_PATH = Path("tokens.json")

def load_tokens() -> Dict[str, str]:
    """Load access/refresh tokens from local json storage."""
    if TOKENS_PATH.exists():
        try:
            data = json.loads(TOKENS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    "access_token": data.get("access_token", "") or "",
                    "refresh_token": data.get("refresh_token", "") or "",
                }
        except Exception:
            pass
    return {"access_token": "", "refresh_token": ""}


def save_tokens(access_token: Optional[str] = None, refresh_token: Optional[str] = None) -> None:
    """Persist tokens to local json storage."""
    data = load_tokens()
    if access_token is not None and access_token != "":
        data["access_token"] = access_token
    if refresh_token is not None and refresh_token != "":
        data["refresh_token"] = refresh_token
    TOKENS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

# Google Gemini configuration
# IMPORTANT: Set GEMINI_API_KEY in environment variable or .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    GEMINI_API_KEY = GEMINI_API_KEY.strip()

try:
    import tiktoken

    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None

try:
    import faiss  # type: ignore

    FAISS_OK = True
except Exception:
    FAISS_OK = False

if not GEMINI_API_KEY:
    print("[WARN] GEMINI_API_KEY missing; embedding/index step will fail without it.")

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_client = genai
    EMBED_MODEL = "models/text-embedding-004"
except Exception as exc:
    print(f"[WARN] Failed to import Google Generative AI SDK: {exc}")
    gemini_client = None
    EMBED_MODEL = None

# ---------------------------------------------------------------------------
# OAuth Automation Functions
# ---------------------------------------------------------------------------


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP server to capture OAuth callback."""

    def do_GET(self):
        global auth_code, callback_received
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        if "code" in query_params:
            auth_code = query_params["code"][0]
            callback_received = True
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""
                <html>
                <head><title>Authorization Successful</title></head>
                <body>
                    <h1>Authorization Successful!</h1>
                    <p>You can close this window now.</p>
                    <p>The authorization code has been captured.</p>
                </body>
                </html>
                """
            )
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Error: No authorization code received</h1></body></html>")

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def start_callback_server():
    """Start local HTTP server to capture OAuth callback."""
    server = HTTPServer(("localhost", 8000), CallbackHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print("[INFO] Callback server started on http://localhost:8000")
    return server


async def automate_oauth_flow():
    """Automate the OAuth authorization flow using Playwright."""
    global auth_code, callback_received

    try:
        from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
    except ImportError:
        print("[ERROR] Playwright is not installed.")
        print("[INFO] Please run: pip install playwright")
        print("[INFO] Then run: python -m playwright install chromium")
        return None

    print("\n" + "=" * 80)
    print("ZOHO OAUTH TOKEN AUTOMATION")
    print("=" * 80)
    print("\n[STEP 1] Starting browser automation...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            print(f"[STEP 2] Opening authorization URL...")
            await page.goto(AUTH_URL, wait_until="networkidle", timeout=60000)

            # Sign in with email and password
            print("[STEP 3] Signing in with credentials...")
            try:
                await page.wait_for_timeout(2000)

                email_selectors = [
                    'input[type="email"]',
                    'input[name="loginId"]',
                    'input[placeholder*="Email"]',
                    'input[placeholder*="email"]',
                    'input[id*="email"]',
                    'input[id*="login"]',
                ]

                email_filled = False
                for selector in email_selectors:
                    try:
                        email_field = page.locator(selector).first
                        if await email_field.is_visible(timeout=3000):
                            await email_field.fill(ZOHO_EMAIL)
                            print(f"[OK] Email entered")
                            email_filled = True
                            break
                    except Exception:
                        continue

                if not email_filled:
                    print("[WARN] Could not find email field automatically.")
                    await page.wait_for_timeout(5000)

                await page.wait_for_timeout(1000)
                next_selectors = [
                    'button:has-text("Next")',
                    'button[type="submit"]',
                    'input[type="submit"]',
                    'button:has-text("Sign in")',
                ]

                next_clicked = False
                for selector in next_selectors:
                    try:
                        next_button = page.locator(selector).first
                        if await next_button.is_visible(timeout=3000):
                            await next_button.click()
                            print(f"[OK] Clicked Next button")
                            next_clicked = True
                            break
                    except Exception:
                        continue

                await page.wait_for_timeout(2000)

                password_selectors = [
                    'input[type="password"]',
                    'input[name="password"]',
                    'input[id*="password"]',
                    'input[id*="passwd"]',
                ]

                password_filled = False
                for selector in password_selectors:
                    try:
                        password_field = page.locator(selector).first
                        if await password_field.is_visible(timeout=5000):
                            await password_field.fill(ZOHO_PASSWORD)
                            print(f"[OK] Password entered")
                            password_filled = True
                            break
                    except Exception:
                        continue

                if not password_filled:
                    print("[WARN] Could not find password field automatically.")
                    await page.wait_for_timeout(5000)

                await page.wait_for_timeout(1000)
                signin_selectors = [
                    'button:has-text("Sign in")',
                    'button:has-text("Sign In")',
                    'button[type="submit"]',
                    'input[type="submit"]',
                    'button:has-text("Next")',
                ]

                signin_clicked = False
                for selector in signin_selectors:
                    try:
                        signin_button = page.locator(selector).first
                        if await signin_button.is_visible(timeout=3000):
                            await signin_button.click()
                            print(f"[OK] Clicked Sign in button")
                            signin_clicked = True
                            break
                    except Exception:
                        continue

                await page.wait_for_timeout(3000)

            except Exception as e:
                print(f"[WARN] Error during sign-in: {e}")

            # Handle concurrent sessions warning page (click "I Understand")
            print("[STEP 3.5] Checking for concurrent sessions warning...")
            try:
                await page.wait_for_timeout(2000)
                
                # Check if we're on the concurrent sessions page
                page_text = await page.text_content("body") or ""
                
                if "concurrent session" in page_text.lower() or "concurrent sessions" in page_text.lower() or "session limit" in page_text.lower():
                    print("[INFO] Concurrent sessions warning detected. Clicking 'I Understand'...")
                    
                    # Try multiple selectors for "I Understand" button
                    understand_selectors = [
                        'button:has-text("I Understand")',
                        'button:has-text("I UNDERSTAND")',
                        'button:has-text("Understand")',
                        'a:has-text("I Understand")',
                        'button[type="button"]:has-text("Understand")',
                        'button.btn-primary:has-text("Understand")',
                        'button.primary:has-text("Understand")',
                        '[role="button"]:has-text("Understand")',
                        '.btn:has-text("Understand")',
                        'button[id*="understand"]',
                        'button[class*="understand"]',
                    ]
                    
                    understood = False
                    for selector in understand_selectors:
                        try:
                            understand_button = page.locator(selector).first
                            if await understand_button.is_visible(timeout=3000):
                                await understand_button.click()
                                print("[OK] Clicked 'I Understand' on concurrent sessions page")
                                understood = True
                                await page.wait_for_timeout(2000)
                                break
                        except Exception:
                            continue
                    
                    if not understood:
                        # Try finding by text content and clicking parent button
                        try:
                            understand_text = page.locator('text="I Understand"').first
                            if await understand_text.is_visible(timeout=3000):
                                # Try to find parent button element
                                parent_button = understand_text.locator('..').first
                                if await parent_button.is_visible(timeout=2000):
                                    await parent_button.click()
                                    print("[OK] Clicked 'I Understand' (via text parent)")
                                    understood = True
                                    await page.wait_for_timeout(2000)
                        except Exception:
                            pass
                    
                    if understood:
                        print("[OK] Proceeding past concurrent sessions warning...")
                        await page.wait_for_timeout(2000)
                    else:
                        print("[WARN] Could not find 'I Understand' button. Continuing anyway...")
                        await page.wait_for_timeout(2000)
                else:
                    print("[INFO] No concurrent sessions warning detected. Continuing...")
                    
            except Exception as e:
                print(f"[WARN] Error handling concurrent sessions page: {e}")

            # Select CRM PRODUCTION
            print("[STEP 4] Looking for CRM PRODUCTION option...")
            try:
                await page.wait_for_timeout(2000)
                selectors = [
                    'text="CRM PRODUCTION"',
                    'text="CRM Production"',
                    'text="Production"',
                    '[data-testid*="production"]',
                    'button:has-text("CRM")',
                ]

                clicked = False
                for selector in selectors:
                    try:
                        element = page.locator(selector).first
                        if await element.is_visible(timeout=3000):
                            await element.click()
                            print(f"[OK] Clicked on CRM PRODUCTION")
                            clicked = True
                            break
                    except Exception:
                        continue

                if not clicked:
                    print("[WARN] Could not find CRM PRODUCTION button automatically.")
                    await page.wait_for_timeout(5000)

            except Exception as e:
                print(f"[WARN] Error selecting CRM PRODUCTION: {e}")

            # Select "money compound"
            print("[STEP 5] Looking for 'money compound' option...")
            try:
                await page.wait_for_timeout(2000)
                selectors = [
                    'input[type="radio"]:near(text="Money Compound")',
                    'input[type="radio"]:near(text="money compound")',
                    'label:has-text("Money Compound")',
                    'label:has-text("money compound")',
                    'text="Money Compound"',
                    'text="money compound"',
                ]

                clicked = False
                for selector in selectors:
                    try:
                        element = page.locator(selector).first
                        if await element.is_visible(timeout=3000):
                            await element.click()
                            print(f"[OK] Clicked on 'money compound'")
                            clicked = True
                            break
                    except Exception:
                        continue

                if not clicked:
                    print("[WARN] Could not find 'money compound' option automatically.")
                    await page.wait_for_timeout(5000)

            except Exception as e:
                print(f"[WARN] Error selecting money compound: {e}")

            # Click Submit button
            print("[STEP 5.5] Clicking Submit button...")
            try:
                await page.wait_for_timeout(1000)
                submit_selectors = [
                    'button:has-text("Submit")',
                    'button[type="submit"]',
                    'input[type="submit"]',
                ]

                submit_clicked = False
                for selector in submit_selectors:
                    try:
                        submit_button = page.locator(selector).first
                        if await submit_button.is_visible(timeout=3000):
                            await submit_button.click()
                            print(f"[OK] Clicked Submit button")
                            submit_clicked = True
                            break
                    except Exception:
                        continue

                await page.wait_for_timeout(2000)

            except Exception as e:
                print(f"[WARN] Error clicking Submit: {e}")

            # Check "I allow WealthAI" checkbox and Accept
            print("[STEP 6] Waiting for permissions page...")
            await page.wait_for_timeout(3000)

            print("[STEP 6.1] Checking 'I allow WealthAI' checkbox...")
            try:
                checkbox_selectors = [
                    'input[type="checkbox"]:near(text="I allow WealthAI")',
                    'input[type="checkbox"]:near(text="WealthAI")',
                    'label:has-text("I allow WealthAI")',
                    'input[type="checkbox"]',
                ]

                checkbox_checked = False
                for selector in checkbox_selectors:
                    try:
                        checkbox = page.locator(selector).first
                        if await checkbox.is_visible(timeout=3000):
                            is_checked = await checkbox.is_checked()
                            if not is_checked:
                                await checkbox.check()
                                print(f"[OK] Checked 'I allow WealthAI' checkbox")
                            checkbox_checked = True
                            break
                    except Exception:
                        continue

                if not checkbox_checked:
                    try:
                        wealthai_text = page.locator('text="I allow WealthAI"').first
                        if await wealthai_text.is_visible(timeout=3000):
                            checkbox = page.locator('input[type="checkbox"]').first
                            if await checkbox.is_visible(timeout=2000):
                                is_checked = await checkbox.is_checked()
                                if not is_checked:
                                    await checkbox.check()
                                    print("[OK] Checked checkbox")
                                checkbox_checked = True
                    except Exception:
                        pass

            except Exception as e:
                print(f"[WARN] Error checking checkbox: {e}")

            # Click Accept button
            print("[STEP 6.2] Clicking Accept button...")
            await page.wait_for_timeout(1000)

            accept_selectors = [
                'button:has-text("Accept")',
                'button:has-text("Allow")',
                'button:has-text("Authorize")',
                'button[type="submit"]',
            ]

            accepted = False
            for selector in accept_selectors:
                try:
                    element = page.locator(selector).first
                    if await element.is_visible(timeout=5000):
                        await element.click()
                        print(f"[OK] Clicked Accept button")
                        accepted = True
                        break
                except Exception:
                    continue

            if not accepted:
                print("[WARN] Could not find Accept button automatically.")
                await page.wait_for_timeout(10000)

            # Wait for callback
            print("[STEP 7] Waiting for callback redirect...")
            try:
                await page.wait_for_url(
                    lambda url: "localhost:8000" in url or "zoho/callback" in url,
                    timeout=30000,
                )
                print("[OK] Callback received!")
                current_url = page.url

                if "code=" in current_url:
                    match = re.search(r"code=([^&]+)", current_url)
                    if match:
                        auth_code = match.group(1)
                        callback_received = True
                        print(f"[OK] Authorization code extracted from URL")

            except PlaywrightTimeoutError:
                print("[WARN] Timeout waiting for callback. Checking if code is in current URL...")
                current_url = page.url
                if "code=" in current_url:
                    match = re.search(r"code=([^&]+)", current_url)
                    if match:
                        auth_code = match.group(1)
                        callback_received = True
                        print(f"[OK] Authorization code found in URL: {auth_code[:20]}...")

            await page.wait_for_timeout(2000)

        except Exception as e:
            print(f"[ERROR] Error during automation: {e}")
            import traceback

            traceback.print_exc()

        finally:
            await page.wait_for_timeout(2000)
            await browser.close()

    return auth_code


def exchange_code_for_tokens(code: str):
    """Exchange authorization code for access and refresh tokens."""
    print("\n[STEP 8] Exchanging authorization code for tokens...")

    payload = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "code": code,
    }

    try:
        response = requests.post(TOKEN_URL, data=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")

        if not access_token:
            print(f"[ERROR] No access_token in response: {data}")
            return None, None

        if not refresh_token:
            print(f"[WARN] No refresh_token in response. Response: {data}")

        print("[OK] Tokens received successfully!")
        print(f"[INFO] Access Token: {access_token[:30]}...")
        if refresh_token:
            print(f"[INFO] Refresh Token: {refresh_token[:30]}...")

        return access_token, refresh_token

    except requests.RequestException as e:
        print(f"[ERROR] Failed to exchange code for tokens: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"[ERROR] Response: {e.response.text}")
        return None, None


def save_tokens_after_exchange(access_token: str, refresh_token: Optional[str]) -> None:
    """Persist newly obtained tokens to local storage."""
    print("\n[STEP 9] Saving tokens to tokens.json ...")
    save_tokens(access_token=access_token, refresh_token=refresh_token or None)


async def generate_tokens():
    """Main function to orchestrate the OAuth automation."""
    global auth_code

    server = start_callback_server()
    time.sleep(1)

    try:
        code = await automate_oauth_flow()
        time.sleep(2)

        if not code and auth_code:
            code = auth_code

        if not code:
            print("\n[ERROR] Failed to obtain authorization code.")
            return None, None

        print(f"\n[OK] Authorization code obtained: {code[:20]}...")

        access_token, refresh_token = exchange_code_for_tokens(code)

        if not access_token:
            print("\n[ERROR] Failed to obtain tokens.")
            return None, None

        save_tokens_after_exchange(access_token, refresh_token or "")

        print("\n" + "=" * 80)
        print("[SUCCESS] Token generation complete!")
        print("=" * 80)

        return access_token, refresh_token

    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user.")
        return None, None
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return None, None
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# RAG Pipeline Functions
# ---------------------------------------------------------------------------


def cleanup_old_data() -> None:
    """Delete existing client artifacts."""
    print("\nCleaning up old client data...")
    if ARTIFACTS_ROOT.exists():
        count = 0
        for client_dir in ARTIFACTS_ROOT.iterdir():
            if client_dir.is_dir():
                shutil.rmtree(client_dir)
                count += 1
        print(f"[OK] Removed {count} old client directories")
    else:
        print("[OK] No old data to clean up")
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)


def refresh_access_token() -> Optional[str]:
    """Refresh Zoho access token."""
    tokens = load_tokens()
    refresh_token = tokens.get("refresh_token", "")
    if not (refresh_token and CLIENT_ID and CLIENT_SECRET):
        print("[WARN] Missing refresh credentials; cannot auto-refresh access token.")
        return None

    token_url = f"{ACCOUNTS_DOMAIN}/oauth/v2/token"
    payload = {
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
    }

    try:
        response = requests.post(token_url, data=payload, timeout=30)
    except requests.RequestException as exc:
        print(f"[ERROR] Failed to refresh access token: {exc}")
        return None

    if response.status_code != 200:
        print(f"[ERROR] Refresh token request failed: {response.status_code} - {response.text}")
        return None

    data = response.json()
    new_token = data.get("access_token")
    if new_token:
        save_tokens(access_token=new_token)
        print("[OK] Refreshed Zoho access token.")
    else:
        print(f"[ERROR] Refresh response missing access_token: {data}")
    return new_token


def fetch_all_contacts() -> List[dict]:
    """Fetch all contacts from Zoho CRM."""
    url = f"{API_DOMAIN}/crm/v2/Contacts"
    tokens = load_tokens()
    headers = {"Authorization": f"Zoho-oauthtoken {tokens.get('access_token','')}"}

    all_contacts: List[dict] = []
    page = 1
    per_page = 200

    while True:
        params = {"page": page, "per_page": per_page}
        response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code != 200:
            print(f"Error fetching contacts: {response.status_code} - {response.text[:200]}")
            if response.status_code in (401, 403):
                new_token = refresh_access_token()
                if new_token:
                    headers["Authorization"] = f"Zoho-oauthtoken {new_token}"
                    print("[INFO] Retrying contact fetch with refreshed token...")
                    continue
            break

        data = response.json()
        contacts = data.get("data", [])

        if not contacts:
            break

        all_contacts.extend(contacts)
        print(f"Fetched page {page}: {len(contacts)} contacts")

        if len(contacts) < per_page:
            break

        page += 1

    return all_contacts


def create_enhanced_contact_file(contact: dict, email: str) -> str:
    """Format Zoho contact fields into a human-readable text dossier."""
    account_no = contact.get("Account_No_1", contact.get("Account_No", "N/A"))
    account_name = contact.get("Account_Name", "N/A")
    lead_source = contact.get("Lead_Source", "N/A")
    aum = contact.get("AUM_Rs_Lakhs", "N/A")
    location = contact.get("Location_Office", contact.get("Location_Home", "N/A"))
    company_type = contact.get("Company_Type", "N/A")
    company_name = contact.get("Company_Name", "N/A")
    department = contact.get("Department", "N/A")

    secondary_email = contact.get("Secondary_Email", "N/A")
    pan_no = contact.get("PAN_Number", contact.get("PAN_NO", contact.get("HUF_Pan_Number", "N/A")))
    dob = contact.get("Date_of_Birth_Actual", contact.get("Date_of_Birth", "N/A"))
    anniversary = contact.get("Anniversary_Date", "N/A")
    aadhaar = contact.get("Aadhaar_No", "N/A")

    mailing_street = contact.get("Mailing_Street", "N/A")
    mailing_city = contact.get("Mailing_City", contact.get("City_Name", "N/A"))
    mailing_state = contact.get("Mailing_State", contact.get("State_UT", "N/A"))
    mailing_country = contact.get("Mailing_Country", "N/A")
    mailing_zip = contact.get("Mailing_Zip", contact.get("Pin_Code", "N/A"))

    account_no_2 = contact.get("Account_No_2", "N/A")
    bank_name_2 = contact.get("Bank_Name_2", "N/A")
    ifsc_2 = contact.get("IFSC_Code_2", "N/A")
    micr_2 = contact.get("MICR_2", "N/A")
    annual_income_huf = contact.get("Annual_Income_HUF", "N/A")

    incorp_date = contact.get("Date_of_Incorporation", "N/A")
    incorp_place = contact.get("Place_of_Incorporation", "N/A")
    assigned_rm = contact.get("Assigned_RM", "N/A")

    first_visit = contact.get("First_Visited_Time", "N/A")
    days_visited = contact.get("Days_Visited", "N/A")
    num_chats = contact.get("Number_Of_Chats", "N/A")
    avg_time = contact.get("Average_Time_Spent_Minutes", "N/A")

    upi_id = contact.get("UPI_id", "N/A")
    comm_address = contact.get("Communication_Address", "N/A")
    account_type = contact.get("Account_Type", "N/A")
    bank_type = contact.get("Bank_Type1", contact.get("Bank_Type", "N/A"))
    category = contact.get("Category_of_Client", "N/A")
    best_time = contact.get("Best_Time_to_Call", "N/A")
    feedback = contact.get("Feedback_Remarks", "N/A")

    content = f"""Enhanced Client Information
============================

BASIC CONTACT DETAILS
---------------------
Name: {contact.get('Full_Name', 'N/A')}
Email: {email}
Secondary Email: {secondary_email}
Phone: {contact.get('Phone', 'N/A')}
Mobile: {contact.get('Mobile', 'N/A')}

IDENTITY & VERIFICATION
-----------------------
PAN Number: {pan_no}
Aadhaar Number: {aadhaar}
Date of Birth: {dob}
Anniversary Date: {anniversary}

MAILING ADDRESS
--------------
Street: {mailing_street}
City: {mailing_city}
State: {mailing_state}
Country: {mailing_country}
ZIP Code: {mailing_zip}

PRIMARY BANK ACCOUNT
-------------------
Account Number: {account_no}
Account Name: {account_name}
Bank Name: {contact.get('Bank_Name', 'N/A')}
IFSC Code: {contact.get('IFSC_Code', 'N/A')}
MICR: {contact.get('MICR', 'N/A')}
Account Type: {account_type}
Bank Type: {bank_type}
Annual Income: {contact.get('Annual_Income', 'N/A')}

SECONDARY BANK ACCOUNT
---------------------
Account Number 2: {account_no_2}
Bank Name 2: {bank_name_2}
IFSC Code 2: {ifsc_2}
MICR 2: {micr_2}
Annual Income HUF: {annual_income_huf}

BUSINESS INFORMATION
-------------------
Company Name: {company_name}
Company Type: {company_type}
Department: {department}
Date of Incorporation: {incorp_date}
Place of Incorporation: {incorp_place}
Assigned RM: {assigned_rm}
Category of Client: {category}

ASSETS & INVESTMENTS
-------------------
AUM (Assets Under Management): {aum} Lakhs
Office Location: {location}

SIP & MUTUAL FUND INFORMATION
-----------------------------
SIP TopUp Date: {contact.get('SIP_TopUp_Date', 'N/A')}
SIP TopUp Amount: {contact.get('SIP_TopUp', 'N/A')}
SIP Alarm Date: {contact.get('SIP_Alarm_date', 'N/A')}

LEAD & SALES INFORMATION
-----------------------
Lead Source: {lead_source}
Lead Source Name: {contact.get('Lead_Source_Name', 'N/A')}

ENGAGEMENT & ACTIVITY
--------------------
First Visit Time: {first_visit}
Days Visited: {days_visited}
Number of Chats: {num_chats}
Average Time Spent: {avg_time} minutes
Communication Address: {comm_address}

PAYMENT & PREFERENCES
--------------------
UPI ID: {upi_id}
Best Time to Call: {best_time}
Feedback Remarks: {feedback}

ADDITIONAL FIELDS
----------------"""

    excluded_fields = {
        "Full_Name", "Email", "Phone", "Mobile", "Secondary_Email", "Account_Name",
        "Account_No_1", "Account_No", "Account_No_2", "Company_Name", "Company_Type",
        "Department", "AUM_Rs_Lakhs", "Location_Office", "Location_Home", "Lead_Source",
        "Lead_Source_Name", "SIP_TopUp_Date", "SIP_TopUp", "SIP_Alarm_date", "PAN_Number",
        "PAN_NO", "HUF_Pan_Number", "Aadhaar_No", "Date_of_Birth", "Date_of_Birth_Actual",
        "Anniversary_Date", "Mailing_Street", "Mailing_City", "Mailing_State", "Mailing_Country",
        "Mailing_Zip", "Bank_Name", "Bank_Name_2", "IFSC_Code", "IFSC_Code_2", "MICR", "MICR_2",
        "Account_Type", "Bank_Type", "Bank_Type1", "Annual_Income", "Annual_Income_HUF",
        "Date_of_Incorporation", "Place_of_Incorporation", "Assigned_RM", "Category_of_Client",
        "First_Visited_Time", "Days_Visited", "Number_Of_Chats", "Average_Time_Spent_Minutes",
        "Communication_Address", "UPI_id", "Best_Time_to_Call", "Feedback_Remarks", "id",
        "Created_Time", "Modified_Time", "Owner", "$approval", "$field_states", "$locked_for_me",
        "$process_flow", "$state", "City_Name", "State_UT", "Pin_Code",
    }

    additional_fields = []
    for key, value in contact.items():
        if (
            value
            and str(value).strip()
            and key not in excluded_fields
            and not isinstance(value, (dict, list))
        ):
            additional_fields.append(f"{key}: {value}")

    if additional_fields:
        content += "\n" + "\n".join(additional_fields[:20])
    else:
        content += "\nNo additional fields with data"

    content += f"""

TECHNICAL DETAILS
----------------
Contact ID: {contact.get('id', 'N/A')}
Created: {contact.get('Created_Time', 'N/A')}
Modified: {contact.get('Modified_Time', 'N/A')}
Owner: {contact.get('Owner', {}).get('name', 'N/A') if contact.get('Owner') else 'N/A'}

Additional Notes:
- This enhanced data was extracted from Zoho CRM
- Includes business, financial, and investment information
- Last updated: {contact.get('Modified_Time', 'N/A')}
"""

    return content


def fetch_related_records(contact_id: str, related_module: str = "Deals") -> List[dict]:
    """Fetch related records (like Deals, Portfolios, etc.) for a contact."""
    tokens = load_tokens()
    headers = {"Authorization": f"Zoho-oauthtoken {tokens.get('access_token','')}"}
    
    # Try different API endpoints for related records
    urls = [
        f"{API_DOMAIN}/crm/v2/Contacts/{contact_id}/{related_module}",  # Standard related list
        f"{API_DOMAIN}/crm/v2/{related_module}",  # Direct module with contact filter
    ]
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                records = data.get("data", [])
                if records:
                    # Filter records related to this contact
                    filtered = []
                    for record in records:
                        # Check if record has contact reference
                        owner = record.get("Owner", {})
                        if isinstance(owner, dict):
                            owner_id = owner.get("id", "")
                        else:
                            owner_id = str(owner)
                        
                        # Include if contact is owner or if record has Contact_Name/Contact linking
                        if (owner_id == contact_id or 
                            contact_id in str(record.get("Contact_Name", "")) or
                            contact_id in str(record.get("Contact", ""))):
                            filtered.append(record)
                    
                    # If we have any records, return them (even if not filtered perfectly)
                    return filtered if filtered else records
            elif response.status_code in (401, 403):
                new_token = refresh_access_token()
                if new_token:
                    headers["Authorization"] = f"Zoho-oauthtoken {new_token}"
                    continue
            elif response.status_code == 404:
                # Module doesn't exist or wrong endpoint, try next
                continue
        except Exception as e:
            # Silently continue to next URL or method
            continue
    
    # Try fetching with COQL (Zoho Query Language) if available
    try:
        coql_url = f"{API_DOMAIN}/crm/v2/coql"
        query = f"select * from {related_module} where Contact_Name.id = {contact_id} limit 10"
        payload = {"select_query": query}
        response = requests.post(coql_url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
    except Exception:
        pass
    
    return []


def fetch_portfolio_data(contact: dict) -> str:
    """Fetch and format portfolio-related data for a contact."""
    contact_id = contact.get("id")
    if not contact_id:
        return ""
    
    portfolio_sections = []
    
    # Fetch Deals (often contain investment/portfolio information)
    deals = fetch_related_records(contact_id, "Deals")
    if deals:
        portfolio_sections.append("=== DEALS / INVESTMENT DEALS ===")
        portfolio_sections.append("These deals may contain investment, portfolio, or financial product information.\n")
        
        for deal in deals[:10]:  # Limit to first 10 deals
            deal_name = deal.get("Deal_Name", "")
            amount = deal.get("Amount", deal.get("Deal_Amount", ""))
            stage = deal.get("Stage", "")
            closing_date = deal.get("Closing_Date", "")
            description = deal.get("Description", "")
            
            # Skip deals with no meaningful data
            if not any([deal_name, amount, stage, closing_date, description]):
                continue
            
            deal_info = ""
            if deal_name:
                deal_info += f"Deal Name: {deal_name}\n"
            if amount and str(amount).strip() and str(amount).lower() not in ["none", "n/a", ""]:
                deal_info += f"Amount: {amount}\n"
            if stage:
                deal_info += f"Stage: {stage}\n"
            if closing_date:
                deal_info += f"Closing Date: {closing_date}\n"
            if description and str(description).strip():
                deal_info += f"Description: {description[:300]}\n"
            
            # Include ALL deal fields that have values (to capture any portfolio data)
            excluded_keys = {"Deal_Name", "Amount", "Deal_Amount", "Stage", "Closing_Date", "Description", "id", "Owner", "Created_Time", "Modified_Time"}
            deal_fields = []
            for key, value in deal.items():
                if key not in excluded_keys:
                    if value is not None and str(value).strip() and str(value).lower() not in ["none", "n/a", "", "[]"]:
                        if not isinstance(value, (dict, list)):
                            deal_fields.append(f"{key}: {value}")
            
            if deal_fields:
                deal_info += "\nAdditional Deal Information:\n"
                deal_info += "\n".join(f"  {field}" for field in deal_fields)
            
            if deal_info.strip():
                portfolio_sections.append(deal_info.strip())
                portfolio_sections.append("")  # Empty line between deals
    
    # Try to fetch Products (investment products) - but filter out catalog items
    products = fetch_related_records(contact_id, "Products")
    if products:
        # Filter products - only include those with actual client data (quantities, holdings, investments)
        actual_holdings = []
        for product in products:
            product_name = product.get("Product_Name", "")
            quantity = product.get("Quantity", product.get("Holding_Quantity", product.get("Units", "")))
            list_price = product.get("List_Price", product.get("Unit_Price", product.get("Current_Price", "")))
            total_value = product.get("Total", product.get("Total_Value", product.get("Investment_Amount", "")))
            
            # Skip generic catalog items - only include if there's actual client investment data
            has_holding_data = (
                (quantity and str(quantity).strip().lower() not in ["none", "n/a", "", "0"]) or
                (total_value and str(total_value).strip().lower() not in ["none", "n/a", ""]) or
                any(key.lower() in ["holding", "investment", "folio", "units", "quantity", "purchase_price", "current_value"] 
                    for key, val in product.items() 
                    if val and str(val).strip().lower() not in ["none", "n/a", "", "[]"])
            )
            
            if has_holding_data:
                actual_holdings.append(product)
        
        if actual_holdings:
            portfolio_sections.append("\n=== INVESTMENT PRODUCTS / HOLDINGS ===")
            portfolio_sections.append("Client's actual investment holdings and products.\n")
            
            for product in actual_holdings[:15]:  # Limit to first 15 actual holdings
                product_name = product.get("Product_Name", "")
                quantity = product.get("Quantity", product.get("Holding_Quantity", product.get("Units", "")))
                list_price = product.get("List_Price", product.get("Unit_Price", product.get("Current_Price", "")))
                total_value = product.get("Total", product.get("Total_Value", product.get("Investment_Amount", "")))
                
                product_info = ""
                if product_name:
                    product_info += f"Product/Investment: {product_name}\n"
                if quantity and str(quantity).strip().lower() not in ["none", "n/a", ""]:
                    product_info += f"  Quantity/Units: {quantity}\n"
                if list_price and str(list_price).strip().lower() not in ["none", "n/a", ""]:
                    product_info += f"  Price per Unit: {list_price}\n"
                if total_value and str(total_value).strip().lower() not in ["none", "n/a", ""]:
                    product_info += f"  Total Value: {total_value}\n"
                
                # Include ALL product fields with values (to capture portfolio details)
                excluded_keys = {"Product_Name", "Quantity", "Holding_Quantity", "Units", "List_Price", "Unit_Price", "Current_Price", "Total", "Total_Value", "Investment_Amount", "id", "Owner", "Created_Time", "Modified_Time"}
                product_fields = []
                for key, value in product.items():
                    if key not in excluded_keys:
                        if value is not None and str(value).strip() and str(value).lower() not in ["none", "n/a", "", "[]"]:
                            if not isinstance(value, (dict, list)):
                                product_fields.append(f"{key}: {value}")
                
                if product_fields:
                    product_info += "\nAdditional Product Information:\n"
                    product_info += "\n".join(f"  {field}" for field in product_fields)
                
                if product_info.strip():
                    portfolio_sections.append(product_info.strip())
                    portfolio_sections.append("")  # Empty line between products
    
    # Try fetching from custom modules that might contain portfolio data
    # Common module names for portfolios
    portfolio_modules = ["Portfolios", "Investments", "Holdings", "Portfolio_Details", "Investment_Details"]
    for module_name in portfolio_modules:
        records = fetch_related_records(contact_id, module_name)
        if records:
            portfolio_sections.append(f"\n=== {module_name.upper()} ===")
            for record in records[:10]:
                record_info = ""
                for key, value in record.items():
                    if value and str(value).strip() and not isinstance(value, (dict, list)):
                        if key != "id":
                            record_info += f"{key}: {value}\n"
                if record_info:
                    portfolio_sections.append(record_info)
    
    # Extract portfolio-related fields directly from contact
    portfolio_fields = []
    portfolio_keywords = ["portfolio", "investment", "fund", "sip", "mutual", "stock", "equity", 
                          "debt", "asset", "holding", "aum", "allocation", "allocation_percentage",
                          "invest", "securities", "bonds", "commodity", "etf", "ulip", "pms"]
    
    # Check for AUM first (most important portfolio metric)
    aum = contact.get("AUM_Rs_Lakhs") or contact.get("AUM") or contact.get("Assets_Under_Management") or contact.get("Total_AUM")
    if aum and str(aum).strip().lower() not in ["none", "n/a", "", "null", "[]"]:
        portfolio_fields.append(f"Total Portfolio Value (AUM): {aum} Lakhs")
    
    # Check for SIP information
    sip_date = contact.get("SIP_TopUp_Date") or contact.get("SIP_Date")
    sip_amount = contact.get("SIP_TopUp") or contact.get("SIP_Amount") or contact.get("SIP_Investment_Amount")
    sip_alarm = contact.get("SIP_Alarm_date") or contact.get("SIP_Alarm_Date")
    
    if (sip_date and str(sip_date).strip().lower() not in ["none", "n/a", "", "null"]) or \
       (sip_amount and str(sip_amount).strip().lower() not in ["none", "n/a", "", "null"]):
        sip_info = "SIP (Systematic Investment Plan) Details:\n"
        if sip_date and str(sip_date).strip().lower() not in ["none", "n/a", "", "null"]:
            sip_info += f"  Top-Up Date: {sip_date}\n"
        if sip_amount and str(sip_amount).strip().lower() not in ["none", "n/a", "", "null"]:
            sip_info += f"  Top-Up Amount: {sip_amount}\n"
        if sip_alarm and str(sip_alarm).strip().lower() not in ["none", "n/a", "", "null"]:
            sip_info += f"  Alarm Date: {sip_alarm}\n"
        portfolio_fields.append(sip_info.strip())
    
    # Extract all portfolio-related fields from contact
    for key, value in contact.items():
        if value is not None and str(value).strip() and not isinstance(value, (dict, list)):
            value_str = str(value).strip().lower()
            if value_str not in ["none", "n/a", "", "null", "[]"]:
                key_lower = key.lower()
                if any(keyword in key_lower for keyword in portfolio_keywords):
                    portfolio_fields.append(f"{key}: {value}")
    
    if portfolio_fields:
        portfolio_header = "=== PORTFOLIO INFORMATION FROM CONTACT ===\n"
        portfolio_header += "This section contains portfolio and investment data directly from the contact record.\n\n"
        portfolio_sections.insert(0, portfolio_header + "\n".join(portfolio_fields))
    
    # Clean up empty sections and check if we have meaningful data
    portfolio_sections = [s for s in portfolio_sections if s and s.strip()]
    
    # Check if we actually have meaningful portfolio data (not just headers)
    has_meaningful_data = False
    if portfolio_fields:  # If we found portfolio fields in contact
        has_meaningful_data = True
    
    # Check if deals or products sections have actual data
    portfolio_text = "\n".join(portfolio_sections)
    if "Deal Name:" in portfolio_text or "Product/Investment:" in portfolio_text:
        has_meaningful_data = True
    
    # If no meaningful portfolio data found, add helpful message
    if not has_meaningful_data and portfolio_sections:
        portfolio_sections.insert(0, "=== PORTFOLIO INFORMATION ===\n")
        portfolio_sections.append(
            "\nNOTE: The deals and products shown above may not contain portfolio/investment data.\n"
            "If portfolio information is missing:\n"
            "- Portfolio data may be stored in custom modules (Portfolios, Investments, Holdings)\n"
            "- Portfolio fields (AUM_Rs_Lakhs, SIP_TopUp, etc.) may be empty in contact records\n"
            "- Investment data may be in related records that aren't properly linked"
        )
    elif not portfolio_sections:
        portfolio_sections = [
            "=== PORTFOLIO INFORMATION ===\n",
            "IMPORTANT: No portfolio data found in Zoho CRM for this contact.\n",
            "This could mean:\n",
            "1. Portfolio data is stored in custom modules (Portfolios, Investments, Holdings, etc.)\n",
            "2. Portfolio information is in related Deals or Products that haven't been linked\n",
            "3. Portfolio fields (AUM, SIP details, etc.) are empty in the contact record\n",
            "4. Portfolio data may need to be fetched using different API endpoints\n\n",
            "To access portfolio data:\n",
            "- Check if custom portfolio modules exist in your Zoho CRM\n",
            "- Verify that deals/products are properly linked to contacts\n",
            "- Ensure portfolio-related fields (AUM_Rs_Lakhs, SIP_TopUp, etc.) are populated in contact records"
        ]
    
    return "\n".join(portfolio_sections) if portfolio_sections else ""


def run_fetch_step() -> None:
    """Perform Zoho extraction and manifest generation."""
    print("=" * 80)
    print("FETCHING ALL CLIENTS FROM ZOHO CRM")
    print("=" * 80)

    cleanup_old_data()

    print("\nFetching contacts from Zoho CRM...")
    contacts = fetch_all_contacts()
    print(f"Total contacts fetched: {len(contacts)}")

    if not contacts:
        print("[ERROR] No contacts fetched. Check your access token or refresh credentials.")
        return

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, List[str]] = {}
    valid_contacts = [
        (contact.get("Email").strip(), contact)
        for contact in contacts
        if contact.get("Email") and contact.get("Email").strip()
    ]

    print(f"Processing {len(valid_contacts)} contacts with valid emails...")

    for idx, (email, contact) in enumerate(valid_contacts, 1):
        client_dir = ARTIFACTS_ROOT / email
        sources_dir = client_dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)

        # Create enhanced contact file
        content = create_enhanced_contact_file(contact, email)
        contact_file = sources_dir / "enhanced_contact_info.txt"
        contact_file.write_text(content, encoding="utf-8")
        manifest[email] = [str(contact_file)]

        # Fetch and add portfolio data
        portfolio_data = fetch_portfolio_data(contact)
        if portfolio_data:
            portfolio_file = sources_dir / "portfolio_information.txt"
            portfolio_file.write_text(portfolio_data, encoding="utf-8")
            manifest[email].append(str(portfolio_file))
            if idx % 50 == 0:
                print(f"  [Portfolio] Found portfolio data for {email}")

        if idx % 50 == 0 or idx == len(valid_contacts):
            print(f"[{idx}/{len(valid_contacts)}] Processed: {email}")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Created enhanced data for {len(valid_contacts)} clients")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print("=" * 80)


def read_txt(path: Path) -> str:
    return path.read_text(errors="ignore")


def file_to_text(path_str: str) -> str:
    p = Path(path_str)
    ext = p.suffix.lower()
    if ext == ".txt":
        return read_txt(p)
    try:
        return p.read_text(errors="ignore")
    except Exception:
        return ""


def tokenize(text: str) -> List[int]:
    if ENC is None:
        return re.findall(r"\S+", text)
    return ENC.encode(text)


def detokenize(tokens: List[int]) -> str:
    if ENC is None:
        return " ".join(tokens)
    return ENC.decode(tokens)


def iter_chunks(text: str, target_tokens: int = TARGET_TOKENS, overlap: int = OVERLAP_TOKENS) -> Iterable[str]:
    toks = tokenize(text)
    if not toks:
        return []
    i = 0
    while i < len(toks):
        j = min(len(toks), i + target_tokens)
        chunk_tokens = toks[i:j]
        yield detokenize(chunk_tokens)
        if j == len(toks):
            break
        i = j - overlap


def build_chunks_for_client(email: str, files: List[str]) -> None:
    client_dir = ARTIFACTS_ROOT / email
    client_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = client_dir / "chunks.jsonl"
    meta_path = client_dir / "chunks_meta.json"

    meta: List[dict] = []
    with chunks_path.open("w", encoding="utf-8") as fout:
        gid = 0
        for file_path in files:
            text = file_to_text(file_path)
            if not text.strip():
                continue
            for idx, chunk in enumerate(iter_chunks(text)):
                record = {"id": gid, "email": email, "source": file_path, "chunk_id": idx, "text": chunk}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta.append(
                    {
                        "id": gid,
                        "email": email,
                        "source": file_path,
                        "chunk_id": idx,
                        "text_preview": chunk[:1200],
                    }
                )
                gid += 1

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[chunks] {email}: {len(meta)} chunks")


def run_chunk_step() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit("clients_manifest.json not found. Run the fetch step first.")

    manifest: Dict[str, List[str]] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"Found manifest with {len(manifest)} clients")

    for email, files in manifest.items():
        print(f"Processing client: {email}")
        files = [f for f in files if Path(f).exists()]
        if not files:
            print(f"[skip] no sources for {email}")
            continue
        print(f"  Found {len(files)} files")
        build_chunks_for_client(email, files)

    print("[done] chunking complete.")


def read_chunks(email: str) -> List[dict]:
    path = ARTIFACTS_ROOT / email / "chunks.jsonl"
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def embed_texts(batch: List[str]) -> np.ndarray:
    if gemini_client is None or EMBED_MODEL is None:
        raise SystemExit("Google Gemini client not configured. Set GEMINI_API_KEY.")
    
    vectors = []
    try:
        # Process embeddings individually (Gemini API handles one at a time for embeddings)
        for text in batch:
            try:
                result = genai.embed_content(
                    model=EMBED_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                vectors.append(result['embedding'])
            except Exception as e:
                error_msg = str(e)
                if "leaked" in error_msg.lower() or "403" in error_msg or "PermissionDenied" in error_msg:
                    print("\n" + "="*80)
                    print("[ERROR] API Key Blocked for Embeddings!")
                    print("="*80)
                    print("Your Google Gemini API key has been flagged as leaked.")
                    print("Even though it might work for chat, the embedding API blocks it.")
                    print("\nSOLUTION: Get a NEW API key:")
                    print("1. Go to: https://makersuite.google.com/app/apikey")
                    print("2. Create a NEW API key")
                    print("3. Update your .env file with the new key")
                    print("4. Restart the server")
                    print("="*80 + "\n")
                    raise SystemExit(1)
                print(f"[WARN] Failed to embed one text: {e}")
                # Add zero vector as fallback
                if vectors:
                    dim = len(vectors[0])
                else:
                    dim = 768  # Default embedding dimension
                vectors.append([0.0] * dim)
    except SystemExit:
        raise
    except Exception as exc:
        message = str(exc)
        if "invalid_api_key" in message or "API key" in message or "401" in message:
            print("[ERROR] Invalid or missing GEMINI_API_KEY.")
            print("        Set a valid key in your environment or .env as GEMINI_API_KEY.")
            raise SystemExit(1)
        raise
    
    arr = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return (arr / norms).astype("float32")


def index_client(email: str) -> None:
    rows = read_chunks(email)
    if not rows:
        print(f"[skip] no chunks for {email}")
        return

    texts = [row["text"] for row in rows]
    meta = [{"id": row["id"], "source": row["source"], "chunk_id": row["chunk_id"]} for row in rows]

    print(f"[embed] {email}: {len(texts)} chunks")
    batches: List[np.ndarray] = []
    batch_size = 256
    for start in range(0, len(texts), batch_size):
        batches.append(embed_texts(texts[start : start + batch_size]))
    embeddings = np.vstack(batches).astype("float32")

    client_dir = ARTIFACTS_ROOT / email
    (client_dir / "embeddings.npy").write_bytes(embeddings.tobytes())
    (client_dir / "chunks_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if FAISS_OK:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(client_dir / "faiss.index"))
        print(f"[index] {email}: faiss.index saved")
    else:
        print("[warn] FAISS not available; numpy search will be used at query time.")

    print(f"[done] {email}")


def run_index_step() -> None:
    print("Starting indexing process...")
    if not ARTIFACTS_ROOT.exists():
        print("[ERROR] No client artifacts found. Run fetch and chunk steps first.")
        return
    client_dirs = [d for d in ARTIFACTS_ROOT.iterdir() if d.is_dir()]
    print(f"Found {len(client_dirs)} client directories")

    for client_dir in client_dirs:
        index_client(client_dir.name)

    print("[done] indexing complete.")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete RAG Pipeline with Automated OAuth Token Generation.")
    parser.add_argument("--skip-token", action="store_true", help="Skip OAuth token generation.")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip Zoho data extraction.")
    parser.add_argument("--skip-chunk", action="store_true", help="Skip chunk generation.")
    parser.add_argument("--skip-index", action="store_true", help="Skip embedding/index creation.")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing client artifacts before fetching new data.",
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Run as a scheduled service that executes daily at 10 PM.",
    )
    parser.add_argument(
        "--schedule-time",
        default="22:00",
        help="Time to run the scheduled job (HH:MM format, default: 22:00 for 10 PM).",
    )
    return parser.parse_args()


async def main_async():
    """Main async function to orchestrate the complete pipeline."""
    args = parse_args()

    # Step 0: Generate tokens if needed
    if not args.skip_token:
        print("\n" + "=" * 80)
        print("STEP 0: OAUTH TOKEN GENERATION")
        print("=" * 80)
        access_token, refresh_token = await generate_tokens()
        if not access_token:
            print("[ERROR] Token generation failed. Exiting.")
            return
        # Tokens are persisted to tokens.json; nothing else to do here.
    else:
        print("=== OAuth token generation skipped ===")

    # Step 1: Fetch data
    if not args.skip_fetch:
        if args.no_clean:
            print("[INFO] Skipping cleanup before fetch per --no-clean flag.")
        else:
            cleanup_old_data()
        run_fetch_step()
    else:
        print("=== Zoho data extraction skipped ===")

    # Step 2: Chunking
    if not args.skip_chunk:
        run_chunk_step()
    else:
        print("=== Chunking skipped ===")

    # Step 3: Indexing
    if not args.skip_index:
        run_index_step()
    else:
        print("=== Embedding/Indexing skipped ===")

    print("\n" + "=" * 80)
    print("[SUCCESS] Complete pipeline finished!")
    print("=" * 80)


async def run_scheduled_pipeline():
    """Run the complete pipeline step by step for scheduled runs."""
    print("\n" + "=" * 80)
    print(f"SCHEDULED PIPELINE RUN - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Step 1: Generate tokens
    print("\n[STEP 1] Generating OAuth tokens...")
    access_token, refresh_token = await generate_tokens()
    if not access_token:
        print("[ERROR] Token generation failed. Exiting.")
        return
    print("[OK] Tokens generated and saved to tokens.json.")
    
    # Step 2: Fetch clients data
    print("\n[STEP 2] Fetching clients data from Zoho CRM...")
    cleanup_old_data()
    run_fetch_step()
    
    # Step 3: Create chunks
    print("\n[STEP 3] Creating chunks from data...")
    run_chunk_step()
    
    # Step 4: Generate embeddings and indexing
    print("\n[STEP 4] Generating embeddings and indexing...")
    run_index_step()
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Scheduled pipeline run completed!")
    print("=" * 80)


def scheduled_job():
    """Wrapper function for scheduled job execution."""
    try:
        asyncio.run(run_scheduled_pipeline())
    except Exception as e:
        print(f"\n[ERROR] Scheduled job failed: {e}")
        import traceback
        traceback.print_exc()


def start_scheduler(schedule_time: str = "22:00"):
    """Start the scheduler to run pipeline daily at specified time."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print("[ERROR] APScheduler is not installed.")
        print("[INFO] Please run: pip install APScheduler")
        sys.exit(1)
    
    # Parse time (HH:MM format)
    try:
        hour, minute = map(int, schedule_time.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("Invalid time format")
    except (ValueError, AttributeError):
        print(f"[ERROR] Invalid time format: {schedule_time}. Use HH:MM format (e.g., 22:00)")
        sys.exit(1)
    
    scheduler = BlockingScheduler()
    
    # Schedule job to run daily at specified time
    scheduler.add_job(
        scheduled_job,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="daily_rag_pipeline",
        name="Daily RAG Pipeline",
        replace_existing=True,
    )
    
    print("\n" + "=" * 80)
    print("SCHEDULER STARTED")
    print("=" * 80)
    print(f"Pipeline will run daily at {schedule_time} (24-hour format)")
    print("\nPipeline steps at each run:")
    print("  1. Generate OAuth tokens")
    print("  2. Fetch clients data from Zoho CRM")
    print("  3. Create chunks from data")
    print("  4. Generate embeddings and create index")
    print("\nPress Ctrl+C to stop the scheduler")
    print("=" * 80 + "\n")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] Scheduler stopped by user.")
        scheduler.shutdown()


def main():
    """Main entry point."""
    args = parse_args()
    
    # If scheduler mode is enabled, start the scheduler
    if args.scheduler:
        start_scheduler(args.schedule_time)
    else:
        # Run pipeline once
        try:
            asyncio.run(main_async())
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user.")
            sys.exit(1)


if __name__ == "__main__":
    main()
