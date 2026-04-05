
# Alaska Permafrost: Methane Hotzone Detection & Sensor Deployment

## Project Overview
This pipeline identifies methane ($CH_4$) hotspots across the Alaskan tundra. As Arctic temperatures rise, thawing permafrost releases ancient carbon. This project uses the **OPTICS** (Ordering Points To Identify the Clustering Structure) algorithm to automate the selection of optimal monitoring sites based on spatial coordinates and emission intensity ($X_{warming}$).

## The "Why": Alaskan Climate Context (2025-2026)
This project addresses specific environmental challenges highlighted in recent Arctic research:

* **The 2025 "Sink-to-Source" Shift:** Recent reports from the Arctic Council confirm that large sections of the Alaskan North Slope have officially transitioned from carbon sinks to net carbon sources.
* **Abrupt Thaw Events (2026):** New data shows that "abrupt thaw" events—where permafrost collapses rapidly into thermokarst lakes are releasing methane at rates 2x higher than previously modeled.
* **Sensor Gap:** Traditional monitoring is sparse. This pipeline provides a data-driven strategy to place sensors where they will capture the most significant "legacy carbon" release.

## Technical Implementation
* **Data Engineering:** Normalizes hierarchical Excel headers and filters for $CH_4$ specific observations.
* **Scaling:** Applies `StandardScaler` to ensure Latitude, Longitude, and Warming Flux ($X_{warming}$) have equal weight in the distance-based model.
* **OPTICS Clustering:** Selected for its ability to handle varying densities and identify spatial outliers (noise) that standard K-Means misses.
* **Persistence:** Serializes the `StandardScaler` and `OPTICS` objects via **Pickle** for production-ready inference.
* **Evaluation:** Validates cluster quality using **Silhouette Score** (cohesion) and **Davies-Bouldin Index** (separation).

## Results
* **Silhouette Score:** ~0.69 (Indicates strong, distinct clustering of emission zones).
* **Davies-Bouldin Index:** ~0.39 (Indicates well-separated deployment zones).
<img width="546" height="389" alt="download (1)" src="https://github.com/user-attachments/assets/cf9325cd-f8ee-4259-a801-cda2a02868e6" />
<img width="757" height="583" alt="download" src="https://github.com/user-attachments/assets/7c42c36c-246b-4867-86a6-7c1d3f588423" />

## Repository Structure
* `permafrost_mle.py`: Main processing and modeling script.
* `permafrost_scaler.pkl`: Saved scaling parameters for new data.
* `permafrost_optics_model.pkl`: The trained clustering model.


