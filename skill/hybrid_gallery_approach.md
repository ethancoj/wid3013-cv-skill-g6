# Hybrid Gallery Approach

This document defines the multi-layered navigation and presentation strategy for the Document Restoration Skill.

## 1. The User Journey

The system uses a three-tier navigation model to move the user from broad discovery to granular analysis.

`Chat` $\rightarrow$ `Master Gallery (with Sidebar)` $\rightarrow$ `Filtered Gallery` $\rightarrow$ `HTML Dashboard`

## 2. Tier 1: The Master Gallery (The Entry Point)

The Master Gallery is the primary interface when the user "opens the vault."

* **Purpose:** High-level overview and discovery.
* **Default View:** Displays all documents across all categories, sorted by `timestamp` (newest first).
* **The Sidebar (The Switchboard):** A persistent navigation element on the left side of the screen.
    * **Category List:** A clickable list of all existing categories (pulled from `vault_index.json`).
    * **Special Folders:** A dedicated, highlighted section for the `[Uncategorized]` bucket.
* **Global Controls:**
    * **Global Search:** Search across the entire vault.
    * **Global Sort:** Sort the entire view by Health Score, Name, or Date.

## 3. Tier 2: The Filtered Gallery (The Skimming Layer)

When a user selects a category from the sidebar, the interface transitions to the Filtered Gallery.

* **Purpose:** Focused skimming of a specific document set.
* **View:** A responsive grid of **Document Cards**.
* **Document Card Anatomy:**
    * **Thumbnail:** A fast-loading `cleaned.png` preview.
    * **Status Badge:** Color-coded `[✅ Success]`, `[⚠️ Failed]`, or `[⏳ Processing]`.
    * **Health Badge:** A small indicator of the document's `health_score`.
    * **Metadata:** Filename and timestamp.
* **Local Controls:**
    * **Category-Specific Search:** Search only within the active category.
    * **Local Sort/Filter:** Filter by status (e.g., "Show Failed Only") or sort within the category.

## 4. Tier 3: The HTML Dashboard (The Deep Dive)

The final destination for detailed inspection.

* **Purpose:** Comprehensive analysis of a single document.
* **Scope:** One `Document_ID`.
* **Mechanism:** Triggered by clicking a card in the Gallery. It loads the full `metadata.json` (Golden Payload) and all high-resolution assets.
* **Visuals:** Includes the **Masking Slider** (Before/After), **Layout Maps**, and **Extracted Data Tables**.

## 5. Technical Requirements

* **Navigation:** The transition between Gallery and Dashboard must feel seamless (Single Page Application style).
* **Data Loading:** The Gallery must use lightweight metadata (`category_summary.json` and `vault_index.json`) to ensure fast loading and low token usage.
* **State Management:** The sidebar must track the "Active Category" so the user always knows where they are.
