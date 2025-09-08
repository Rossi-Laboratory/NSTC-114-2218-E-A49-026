# Zotero Library for NSTC Project References

This repository contains a curated **BibTeX file (`references.bib`)** under the folder **`zotero library/`**, which includes the key literature for the National Science and Technology Council (NSTC) project.
The references cover the following research domains:

* **Generative AI (GenAI)**
* **3D Generation and Reconstruction**
* **Vision-Language-Action (VLA)**
* **Vision-Language-Navigation (VLN)**
* **Triboelectric Nanogenerators (TENG)**

## 📖 How to Use with Zotero

1. **Download Zotero**

   * Install Zotero from the official website: [https://www.zotero.org/download/](https://www.zotero.org/download/)

2. **Open the BibTeX Library**

   * Launch Zotero.
   * Go to **File → Import...**
   * Select the `references.bib` file located in the `zotero library/` folder of this repository.

3. **Choose Import Options**

   * In the import dialog, select:

     * **Import into new collection** (recommended, to keep the project references organized).
     * Zotero will automatically parse the `.bib` file and create a structured reference library.

4. **Explore the Literature**

   * After import, you will see all the project’s references in Zotero.
   * You can filter, search, and annotate papers on:

     * Generative AI (foundation models, diffusion, transformers)
     * 3D scene reconstruction and generation (NeRF, 3DGS, 4D representation)
     * VLA/VLN frameworks for robotics
     * TENG-related energy harvesting and sensing applications

5. **Use with LaTeX / Overleaf**

   * You may also link this `references.bib` directly in your LaTeX project with:

     ```latex
     \bibliography{zotero library/references}
     \bibliographystyle{ieee_fullname}
     ```
   * This ensures consistent citation formatting across project papers and proposals.

## 📂 File Structure

```
zotero library/
 └── references.bib   # Main reference database for the NSTC project
```

## 🎯 Purpose

This Zotero library centralizes the **review literature** essential to this NSTC project, making it easier for collaborators and students to access, cite, and expand upon prior work in **GenAI, 3D generation, VLA, VLN, and TENG**.
