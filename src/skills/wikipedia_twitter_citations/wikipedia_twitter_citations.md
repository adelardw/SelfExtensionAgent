# Skill: Wikipedia Twitter/X Citation Counter

## Purpose
Counts how many times Twitter/X posts were cited as references on specific English Wikipedia pages, looking at historical revisions.

## When to use
Use this skill when you need to:
- Count Twitter/X (twitter.com, x.com) links/references cited on Wikipedia pages
- Analyze historical versions of Wikipedia pages (specific revision dates)
- Get per-day breakdown of Twitter citations on Wikipedia date pages

## Tools

### `count_twitter_citations_per_day()`
Counts Twitter/X post citations on English Wikipedia pages for each day of August (August 1 through August 31), looking at the last revision of each page from June 2023.

**Input**: None (hardcoded to August days, June 2023 revisions)
**Output**: Formatted per-day breakdown with total counts of Twitter/X references found in the page HTML content at the June 2023 revision.

## Technical approach
- Uses Wikipedia REST API (`en.wikipedia.org/w/api.php`) to find the last revision of each page before July 1, 2023 (i.e., the last revision from June 2023)
- Fetches parsed HTML content at that specific revision
- Counts all `<a>` links pointing to `twitter.com` or `x.com` in the references/citations section
- Reports per-day results