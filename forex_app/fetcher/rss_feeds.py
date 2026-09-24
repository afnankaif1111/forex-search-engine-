"""RSS feed parser for Forex and Financial News."""

import email.utils
import html
import logging
import re
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


def clean_html(raw_html: str) -> str:
    """Strip HTML markup and unescape HTML entities."""
    if not raw_html:
        return ""
    # Strip CDATA tags
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", raw_html, flags=re.DOTALL)
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespaces
    clean = re.sub(r"\s+", " ", clean).strip()
    return html.unescape(clean)


def parse_date(date_str: str) -> str:
    """Parse various date formats into an ISO 8601 string."""
    if not date_str:
        return datetime.utcnow().isoformat()
    try:
        dt = email.utils.parsedate_to_datetime(date_str)
        return dt.isoformat()
    except Exception:
        pass
    try:
        # Try ISO 8601
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.isoformat()
    except Exception:
        return datetime.utcnow().isoformat()


def fetch_rss_feed(feed_name: str, feed_url: str, timeout: float = 8.0) -> list[dict[str, Any]]:
    """Download and parse an RSS or Atom feed into standard article dictionaries."""
    articles = []

    # Permissive SSL context to avoid certificate failures on government/central bank feeds
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(feed_url, headers=DEFAULT_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
            if response.status != 200:
                logger.warning("Feed %s returned status %d", feed_name, response.status)
                return []
            content = response.read()
    except Exception as e:
        logger.warning("Failed to fetch RSS feed %s from %s: %s", feed_name, feed_url, e)
        return []

    try:
        root = ET.fromstring(content)
    except Exception as e:
        # Fallback: sanitize unescaped ampersands and try once more
        try:
            text = content.decode("utf-8", errors="replace")
            sanitized = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", text)
            root = ET.fromstring(sanitized.encode("utf-8"))
        except Exception as e2:
            logger.warning("XML parsing error for feed %s: %s (fallback failed: %s)", feed_name, e, e2)
            return []

    # Check for standard RSS <channel><item>
    channel = root.find("channel")
    if channel is not None:
        items = channel.findall("item")
        for item in items:
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            pub_date_el = item.find("pubDate")
            guid_el = item.find("guid")

            title = clean_html(title_el.text) if title_el is not None and title_el.text else ""
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            summary = clean_html(desc_el.text) if desc_el is not None and desc_el.text else ""
            pub_date = parse_date(pub_date_el.text) if pub_date_el is not None and pub_date_el.text else datetime.utcnow().isoformat()
            guid = guid_el.text.strip() if guid_el is not None and guid_el.text else link or title

            if title:
                articles.append({
                    "guid": guid,
                    "title": title,
                    "summary": summary,
                    "content": summary,
                    "url": link,
                    "source": feed_name,
                    "published_at": pub_date,
                })
        return articles

    # Check for Atom <entry>
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns) or root.findall("entry")
    for entry in entries:
        title_el = entry.find("atom:title", ns) or entry.find("title")
        link_el = entry.find("atom:link", ns) or entry.find("link")
        summary_el = entry.find("atom:summary", ns) or entry.find("summary") or entry.find("atom:content", ns) or entry.find("content")
        updated_el = entry.find("atom:updated", ns) or entry.find("updated") or entry.find("atom:published", ns) or entry.find("published")
        id_el = entry.find("atom:id", ns) or entry.find("id")

        title = clean_html(title_el.text) if title_el is not None and title_el.text else ""
        link = ""
        if link_el is not None:
            link = link_el.attrib.get("href", "") or (link_el.text.strip() if link_el.text else "")
        summary = clean_html(summary_el.text) if summary_el is not None and summary_el.text else ""
        pub_date = parse_date(updated_el.text) if updated_el is not None and updated_el.text else datetime.utcnow().isoformat()
        guid = id_el.text.strip() if id_el is not None and id_el.text else link or title

        if title:
            articles.append({
                "guid": guid,
                "title": title,
                "summary": summary,
                "content": summary,
                "url": link,
                "source": feed_name,
                "published_at": pub_date,
            })

    return articles
