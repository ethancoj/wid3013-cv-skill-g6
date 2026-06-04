# Document Vault Architecture

This document outlines the storage, organization, and categorization logic for the Document Restoration Skill.

## 1. Storage Hierarchy (The Data Lake)

The vault is organized into a hierarchical structure to support categorization and efficient skimming.

```text
/workspace/document_vault/
├── .metadata/                 <-- (Optional) Central index for fast searching
│   └── vault_index.json       <-- Global list of all docs, categories, and paths
│
├── [Category_Name]/           <-- e.g., "Historical_Letters", "Invoices", "Newspapers"
│   ├── category_summary.json  <-- Lightweight cache for the category
│   └── [Document_ID_AlphaNumeric]/
│       ├── metadata.json      <-- The "Golden Payload" for this document
│       ├── original.png       <-- The raw, noisy input
│       ├── cleaned.png        <-- The binarized/denoised version
│       ├── layout.png         <-- The image with bounding-box overlays
│       └── annotated.png     <-- The image with callouts/highlights
│
└── [Uncategorized]/           <-- Default location for new uploads
    └── [Document_ID_AlphaNumeric]/
        └── ...
```

## 2. Categorization Logic

### 2.1 Fuzzy Matching
To reduce user friction, the skill uses fuzzy matching (e.g., Levenshtein distance) to map user-provided category names to existing categories.

* **High Similarity:** If a match is found above a threshold (e.g., 80%), the skill asks for confirmation: *"I found an existing category 'old_message_chain_1930' that looks like what you meant. Should I use that? (Yes/No)"*
* **Low/Medium Similarity:** If no "slam-dunk" match is found, the skill presents the **Top 5 most similar matches** as a menu (e.g., *"Did you mean 1, 2, 3, 4, or 5? Or type 'New' to create a new category"*).
* **No Match:** If no similarity is detected, a new category folder is created.

### 2.2 User-Directed Methods
* **Conversational:** User attaches files and names the category in chat.
* **Folder-as-Category:** User uploads a folder/archive; the folder name is used as the category.

## 3. The Category Summary (`category_summary.json`)

To enable "skimming" without heavy computational or token costs, each category folder contains a `category_summary.json`. This file is updated whenever a document is added or removed.

### Schema
```json
{
  "category_name": "string",
  "last_updated": "ISO-8601 string",
  "document_count": "integer",
  "total_files_size_mb": "float",
  "content_overview": {
    "common_types": ["string"],
    "date_range": {
      "start": "ISO-8601 string",
      "end": "ISO-8601 string"
    }
  },
  "recent_docs": [
    {
      "id": "string",
      "timestamp": "ISO-8601 string",
      "type": "string"
    }
  ]
}
```

## 4. Skimming & Visualization Flow

1.  **Gallery View (Lightweight):** Loads thumbnails and data from `vault_index.json` and `category_summary.json`.
2.  **Deep Dive (Full Intelligence):** On selection, loads the full `metadata.json` (Golden Payload) and high-resolution assets for the selected `Document_ID`, rendering the **HTML Dashboard**.
