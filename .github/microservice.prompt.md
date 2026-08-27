agent: agent
description: Generates a production-ready async FastAPI microservice for Databricks
Generate FastAPI Microservice:

Standards: Follow copilot-instructions.md exactly, paying close attention to the 9-Step Architecture.

Template Structure:

    Import FastAPI, pydantic, and databricks.connect.

    Define Pydantic input schemas with strict validation.

    Create an @app.post("/process") async endpoint.

    Implement a try/except fallback layer (Graceful Degradation) with a 5-second timeout for external API calls.

    Read/Write outputs to Databricks Delta Tables.

    Include a Docker HEALTHCHECK.

Usage: Return the complete, deployable python code for the requested step.