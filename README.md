# HealthBox 🩺🤖

> **Empowering African Healthcare with Sovereign, Explainable, and AccessEdge AI**


---




## 🌍 Executive Summary

**HealthBox** is an offline-first, edge-AI medical diagnostic assistant designed specifically for peripheral health centers in Sub-Saharan Africa. Born from the **UniPod Benin ecosystem** and the **IA Bootcamp Guinea**, the project aims to democratize access to expert-level radiological interpretation in resource-limited settings.

Unlike traditional cloud-dependent AI solutions, HealthBox operates as a compact, network-connected appliance that analyzes medical imaging (starting with chest X-rays) in **under 60 seconds** without requiring internet connectivity. By generating clinician-ready reports in French and providing explainable AI insights, HealthBox reduces diagnostic delays from days to minutes, empowering local technicians and physicians to make life-saving decisions early. The project is committed to **technological sovereignty**, offering its core software and models under an Open Source license to foster trust, collaboration, and local innovation across the African continent.

---

## ⚠️ The Problem: Diagnostic Isolation

### The Crisis in Rural Healthcare
In Sub-Saharan Africa, and particularly in Benin, the healthcare system faces a critical bottleneck known as **"Diagnostic Isolation."** While imaging equipment may exist in peripheral health centers, the expertise required to interpret these images is concentrated in urban metropolises. This disparity creates a dangerous gap in care delivery. According to 2024 WHO data, Benin has only **2.14 physicians per 10,000 inhabitants**, far below the recommended threshold. In rural areas, this ratio is even lower, leaving technicians without immediate specialist support.

### The Cost of Delay
The consequences of this isolation are measured in human lives. Transporting a radiograph to a specialist in a city can take **days to weeks**. For diseases like Tuberculosis, the median diagnostic delay in resource-limited settings ranges from **60 to 90 days**. During this waiting period, contagious patients remain untreated, and critical conditions like pneumothorax or severe pneumonia can become fatal. Tuberculosis alone accounts for **424,000 deaths in Africa annually**. Furthermore, infrastructure constraints such as intermittent internet connectivity and frequent power outages make cloud-based AI solutions impractical for production use in these environments. HealthBox addresses these systemic failures by bringing the expertise to the point of care, regardless of connectivity.

---

## 💡 The Solution: HealthBox

### What is HealthBox?
HealthBox is a secure, local network appliance that integrates seamlessly into existing clinical workflows. It is designed to be **100% offline**, ensuring continuity of care even when the internet is down. The system connects to the health center's local network, allowing any authorized device (computer, tablet, smartphone) to access its services without needing external connectivity. While the initial phase focuses on **chest X-ray analysis** (detecting Tuberculosis, Pneumonia, Pneumothorax, and Cardiac anomalies), the architecture is modular and designed to expand into other specialties like ultrasound and dermatology without changing the hardware infrastructure.

### Clinical Workflow Integration
HealthBox is built around a simple three-step process that respects the daily routine of health workers:
1.  **Acquisition & Input:** After performing an X-ray, the technician transfers the image to HealthBox via the local network and enters minimal clinical data (age, sex, symptoms) through a simplified web interface.
2.  **Automated Analysis:** The system analyzes the image in the background while the patient is still on-site. Within **60 seconds**, it produces a pre-filled diagnostic report in clinical French, highlights regions of interest using heatmaps, and classifies the case by urgency (Green/Orange/Red).
3.  **Validation:** The physician reviews the AI-generated report, validates or modifies it based on their clinical judgment, and archives it in the patient's record. The physician retains full therapeutic decision-making authority.

### Explainability & Trust
A key differentiator for HealthBox is its commitment to **Explainable AI (XAI)**. Unlike black-box systems that output only a probability score, HealthBox uses Vision-Language Models (VLM) to generate narrative reports that explain the reasoning behind a suspicion (e.g., *"Parenchymal opacity in the right upper lobe, compatible with active tuberculosis"*). This transparency builds trust with clinicians and facilitates appropriate decision-making. Additionally, an automated **Critical Alert System** triggers immediate visual and audible notifications for life-threatening conditions, ensuring urgent cases are prioritized instantly.

---

## 🛠 Technical Architecture

### Cascaded AI Pipeline
To balance speed and accuracy on edge hardware, HealthBox employs a **three-tier cascaded architecture**:
1.  **Tier 1 (Fast Triage):** An **EfficientNet-B0** model (INT8 quantized) performs a rapid initial scan in under 10 seconds to filter out normal cases. This ensures that healthy patients are not bottlenecked by heavier processing.
2.  **Tier 2 (Specialized Validation):** For suspicious cases, an ensemble of **ResNet-50 and DenseNet-121** models is activated to refine the diagnosis. This combination leverages the strengths of both architectures to detect global patterns (like cardiomegaly) and fine textures (like nodules).
3.  **Tier 3 (Narrative Generation):** A fine-tuned **BLIP-2 Vision-Language Model** generates the final clinical report in French. This model aligns visual features with clinical text, providing the explainability required for medical adoption.

### Edge Optimization & Hardware
The system is optimized for **Edge AI** deployment. Models undergo **INT8 quantization** and pruning to reduce size by up to 75% while maintaining accuracy, enabling inference on embedded hardware like the **NVIDIA Jetson Orin Nano** or **RDK X5**. The hardware design prioritizes resilience: it features passive cooling for tropical environments, encrypted hot-swappable SSDs for data security, and a battery backup system (2–6 hours autonomy) to handle frequent power outages. The software stack is built on **Python, PyTorch, and Flask**, containerized with Docker for easy deployment and maintenance.

---

## 📊 Data Strategy & Localization

### Two-Phase Development Approach
HealthBox adopts a pragmatic **"Cloud First, Edge Later"** data strategy to ensure technical robustness before field deployment.
*   **Phase 1 (Public Data):** Initial model development and architecture validation are performed using large public datasets (CheXpert, MIMIC-CXR, Shenzhen TB). This allows for rapid iteration on cloud infrastructure without the immediate need for local data collection.
*   **Phase 2 (Local Data):** Parallel to development, a rigorous local data collection process is underway in Benin. The goal is to collect **5,000 to 15,000 annotated chest X-rays** from diverse urban and rural centers. These images are double-read by expert Beninese radiologists to ensure a **Cohen's Kappa ≥ 0.75**, guaranteeing high-quality ground truth for fine-tuning.

### Addressing Bias
A critical component of the data strategy is **representativeness**. The local dataset is curated to include diverse equipment types, demographic groups (age, sex), and geographic locations (urban vs. rural). This ensures the model performs reliably across the varied conditions found in African health centers, mitigating the bias often present in models trained solely on Western data. The project also commits to publishing anonymized subsets of this data under open licenses to contribute to the global scientific community's understanding of African medical imaging.

---

## 🔒 Privacy, Ethics & Compliance

### Regulatory Framework
HealthBox adheres to a strict framework of data protection, combining national regulations with international standards. In Benin, the project complies with **Law n°2017-20** on personal data protection and seeks authorization from the **APDB** (Data Protection Authority). Ethical approval is obtained from the **National Ethics Committee for Health Research (CNERS)**.

### Technical Safeguards
Data sovereignty is a core principle: **patient data never leaves the health center** unless explicitly consented to for research. All data stored on the device is encrypted using **AES-256**, and local network communications are secured via **TLS 1.3**. The system implements role-based access control (technician, physician, admin) and maintains comprehensive audit logs for traceability. For any optional cloud synchronization (e.g., backups), the project targets **HDS (Health Data Hosting)** certification and follows **DICOM PS3.15** standards for anonymization.

### Ethical Principles
The project operates on the principles of **data minimization** and **informed consent**. Patients are clearly informed about the use of AI in their care pathway and retain the right to withdraw their data at any time. The system is designed as a **Clinical Decision Support Tool**, not a standalone diagnostic device, ensuring that final responsibility always remains with the qualified healthcare professional.

---

## 🗓 Development Roadmap

### Strategy: Cloud First, Edge Later
The project follows a **7-month MVP strategy** designed to reduce risk and accelerate validation:
*   **Months 1–3 (Cloud Development):** Models and APIs are developed and tested on cloud infrastructure. This allows for rapid iteration and functional testing with remote partners without the cost of physical hardware.
*   **Months 4–6 (Edge Transition):** Validated models are optimized (quantization, pruning) for embedded hardware. Prototypes are assembled and tested for stability under local conditions (heat, power cuts).
*   **Months 7+ (Production Deployment):** The final 100% offline Edge version is deployed in pilot centers. Cloud connectivity becomes optional, used only for backups or epidemiological reporting.

This approach ensures that a functional MVP is available for demonstration by Month 2, while the final production system is rigorously tested for the constraints of the African context before deployment.

---


## 📈 Performance Metrics (KPIs)

### Diagnostic & Operational Targets
HealthBox defines strict Key Performance Indicators to ensure clinical safety and utility:
*   **Recall (Sensitivity):** ≥ 95% (Prioritizing the detection of all true positive cases to minimize missed diagnoses).
*   **Precision:** ≥ 85% (Balancing sensitivity to prevent alert fatigue among clinicians).
*   **Inference Time:** ≤ 60 seconds (Ensuring diagnosis while the patient is still present).
*   **System Uptime:** ≥ 99% (Critical for reliable care delivery in unstable environments).
*   **Alert Success:** 100% of critical cases (e.g., pneumothorax) must trigger immediate notifications.

### Validation Methodology
Performance is validated on both public benchmarks (for reproducibility) and a held-out local Beninese test set (for contextual relevance). A model is only deployed if it meets all thresholds on the local test set, ensuring it is robust to the specific imaging conditions and pathologies found in the target region.

---


## 📞 Contact & Partners

**Project Lead:** Jonathan Suru  
**Origin:** UniPod Benin / IA Bootcamp Guinea  
**Email:** hello@jonathansuru.me  
**Website:** [jonathansuru.me](https://jonathansuru.me)

### Strategic Partners
*   **Institutional:** Ministry of Health Benin, WHO Africa, Africa CDC.
*   **Academic:** University of Abomey-Calavi (UAC), University of Parakou, UCAD Senegal.
*   **Technology:** NVIDIA (Jetson Program), Hugging Face, emr4all.
*   **Research:** Deep Learning Indaba, AI4D Africa, Masakhane.

---

> ⚠️ **Medical Disclaimer:** HealthBox is a **clinical decision support tool**, not a standalone diagnostic device. All AI-generated outputs must be reviewed and validated by a qualified healthcare professional. Final therapeutic decisions remain the sole responsibility of the treating physician. This tool does not replace clinical judgment, patient history, or comprehensive examination.