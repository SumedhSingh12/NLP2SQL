import os
import logging

import pandas as pd
import openai

import db_utils
import openai_utils

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
openai.api_key = os.environ["OPENAI_API_KEY"]

if __name__ == "__main__":
    logging.info("Loading data...")
    df_location = input("Enter the location of the CSV file to be used as the dataframe: ")
    df = pd.read_csv(df_location)
    logging.info(f"Data Format: {df.shape}")

    db_name = input("Enter the name of the database: ")

    logging.info("Converting to database...")
    database = db_utils.dataframe_to_database(df, db_name)

    fixed_sql_prompt = openai_utils.create_table_definition_prompt(df, db_name)
    logging.info(f"Fixed SQL Prompt: {fixed_sql_prompt}")

    logging.info("Waiting for user input...")
    user_input = openai_utils.user_query_input()
    final_prompt = openai_utils.combine_prompts(fixed_sql_prompt, user_input)
    logging.info(f"Final Prompt: {final_prompt}")

    logging.info("Sending to OpenAI...")
    response = openai_utils.send_to_openai(final_prompt)
    proposed_query = response.choices[0].message.content
    proposed_query_postprocessed = db_utils.handle_response(response)
    logging.info(f"Response obtained. Proposed sql query: {proposed_query_postprocessed}")
    result = db_utils.execute_query(database, proposed_query_postprocessed)
    logging.info(f"Result: {result}")
