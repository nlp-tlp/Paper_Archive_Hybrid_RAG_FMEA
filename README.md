# Impact of Table to Graph Translation on Hybrid Graph-Based RAG of FMEA Spreadsheets

This repository contains the code and dataset accompanying the paper titled "Impact of Table to Graph Translation on Hybrid Graph-Based RAG of FMEA Spreadsheets" by Heidi Leow, Prof. Melinda Hodkiewicz, and Dr. Caitlin Woods.

## Overview

Failure Modes and Effects Analysis (FMEA) is a widely used methodology in risk engineering to identify potential system failures, assess their causes and effects, and prioritise mitigation actions. Data captured in FMEA development is almost universally stored in loosely-structured Excel spreadsheets, that contain longer free-form textual descriptions, categorical/ constrained text, and numeric fields. However, vector search lacks support for structured filtering and aggregation, while existing Text-to-Query pipelines struggle with linguistic variation in descriptive text.

The project presents a hybrid RAG retrieval system that maps graph queries with custom text matching functions (BM25 fuzzy, vector semantic) to standard Neo4j Cypher, and uses it to assess the performance of different graphs constructed from the same FMEA dataset.

This repository contain the following suporting materials:

1. [Hybrid system that can switch between graph constructors of different structures and text matching strategies.](#1-hybrid-system)
2. [Dataset of compiled real FMEA spreadsheets and associated competency questions with model answers.](#2-dataset-and-competency-questions)
3. [Evaluation pipeline to automatically run all competency questions and judge pipeline outputs.](#3-evaluation-pipeline)

## 1. Hybrid System

To enable hybrid retrieval in a single interface, we extend Neo4j’s Cypher query language with two abstract functions, `IS_SEMANTIC_MATCH()` and `IS_FUZZY_MATCH()`. These functions provide a minimal interface for the retriever, invoked within `WHERE` clauses in the same style as standard attribute filtering. At runtime, they are translated into lower-level operations that leverage Neo4j’s native vector similarity and full-text capabilities.

We provide three graph structures based on the FMEA dataset's original tabular schema, as the core axis for comparison:

1. Field Graph: Each textual column in the FMEA graph is given its own entity. All numeric fields are attached to the `FailureMode` entity as properties.
2. Concept Graph: Textual columns that are indistinguishable by content are concatenated together as key-value pairs with `|` delimiters. All numeric fields are attached to the `FailureOccurrence` entity.
3. Row Graph: All textual fields in a row are concatenated together as key-value pairs with `|` delimiters. All numeric fields are attached to the `Row` entity as properties.

The hybrid graph retrieval system must first be configured with the loaded graph translations for interfacing and evaluation to work.

### 1.1 Packages and Configuration

To install required packages for Python, run inside your preferred virtual environment:

```shell
pip install -r requirements.txt
```

A configuration file with the name `.env` should be set up with your details. A template version is located under [.env.template](.env.template). Duplicate and rename this file, and fill in the values. An instance of **Neo4j** needs to be set up and running on your system as a new project. An **OpenAI** key is also needed to access GPT models.

### 1.2 Loading Data

The [load.py](src/load.py) file is set up as a pseudo command-line interface. Calling it with the appropriate arguments will set up graph structures according to the paper.

First, to parse in the data from the spreadsheet run:

```shell
cd src
python3 load.py property_text skb
```

Then, run all or the desired structures from (`property_text`, `concept_text`, `row_text`, `row_all`) replacing `[structure]` below:

```shell
python3 load.py [structure] chroma
python3 load.py [structure] neo4j # except row_all (baseline vector search)
```

Graph structures set up in this way are ready to query in the [chat interface](#2-chat-interface), [evaluation pipeline](#4-evaluation-pipeline) and any other code.

### 1.3 Running Chat Interface

A Streamlit interface is also provided to allow easy access to the configured RAG strategies and vector search collections. To use the interface, run the following command:

```shell
cd src
python3 -m streamlit run app/streamlit_app.py
```

This will automatically open your browser to the app's local address. Alternatively, to run/refresh the interface without starting a new browser tab add the `--server.headless true` flag to the end of the command.

## 2. Dataset and Competency Questions

The FMEA dataset analysed in this study comprises 330 rows, drawn from multiple Excel spreadsheet tables sourced from a mining organisation. The data has been anonymised for proprietary reasons. Each table records failure modes for components within a mining truck system hierarchy. Each row is typically associated with one failure mode, and the tables used for this dataset follow a consistent schema.

This repository also contains a set of competency questions that serve as a benchmark for this study. Each question is categorised by the type of structured and unstructured operations required to answer it. The set comprises 40 questions.

### 2.1 Locating Files

The locations for the dataset and QA set are located under the [data](data/) directory:

1. The [dataset](data/dataset/) subdirectory includes the file [fmea_dataset_filled.csv](data/dataset/fmea_dataset_filled.csv), which is the main data used for loading and experimentation. It also contains the original compiled FMEA table in [fmea_dataset_combined.csv](data/dataset/fmea_dataset_combined.csv).
2. The [questions](data/questions/) subdirectory includes the file [fmea_qa_model.xlsx](data/questions/fmea_qa_model.xlsx), which includes the evaluation questions, model final answers, and associated operation type categories and nuggets. It also includes a set of model retrieval queries for the Field Graph in [fmea_queries_field.xlsx](data/questions/fmea_queries_field.xlsx).

## 3. Evaluation pipeline

We manually extract minimal information units (nuggets) from gold-standard answers and mark them as either essential or optional for an adequate answer. Marking is carried out by comparing system outputs to these nuggets, and annotating whether each are matched, missing, or incorrect. Our Nugget Recall (NR) metric is computed as the proportion of matched essential nuggets, summed with the number of incorrect optional nuggets. LLM-as-a-Judge scoring is available to automate the evaluation process.

### 3.1 Running Experiment

The [evaluate.py](src/evaluate.py) file is set up in a similar way to [load.py](src/load.py). Calling it with the appropriate arguments will run pipelines against all the [competency questions](#3-dataset-and-competency-questions) and/or judge its outputs according to the paper.

To run a RAG strategy defined in [src/scopes/\_\_init\_\_.py](src/scopes/__init__.py) against all the competency questions use:

```shell
cd src
python3 evaluate.py [strategy] rag [allow_linking]
```

Provide any 4th argument to run the entity linking version. The RAG responses will be located under the [src/evaluation/experiment_runs](src/evaluation/experiment_runs/) directory.

### 3.2 Performing Automatic Evaluation

To set up automatic judging with LLM-as-a-Judge, information nuggets are required as a rubric for the model to match against. This is already available in the [model answers Excel file](#3-dataset-and-competency-questions), but this automatic process can be run again with:

```shell
python3 evaluate.py property_descriptive nugget
```

Then, to run do a full experiment run-through for a strategy defined in [src/scopes/\_\_init\_\_.py](src/scopes/__init__.py) use:

```shell
python3 evaluate.py [strategy] rag [allow_linking]
```

Alternatively, to run retrieval and evaluation for all strategies defined at once, use:

```shell
python3 evaluate.py loop_all
```
