# RL Browser Agent

> 🔨 **Note:** This repository is still in development. Contributions and feedback are welcome!

## Setup

- Make sure you have UV installed
- Install requirements: `uv sync`
- Install browser: `uv run python -m playwright install`
  - This project uses Playwright to control the browser. You can install the browser of your choice using the command above.
- Write your environment variables in a `.env` file (see `.env.test`)

## Example

```python
from pydantic import BaseModel
from browser import WebScraper

task = "Find the oldest open issue on the PyTorch repository."

class IssueModel(BaseModel):
    date: str
    title: str
    author: str
    description: str

scraper = WebScraper(task, "https://github.com/pytorch/pytorch", IssueModel, callback=print)
await scraper.run()
```

## Testing

In the works

## Status

- ✅ Basic functionality
- 🛠️ Testing
- 🛠️ Documentation

### Stack

- Browser: [Playwright](https://github.com/microsoft/playwright-python)

## Alternatives

- https://github.com/CognosysAI/browser/
