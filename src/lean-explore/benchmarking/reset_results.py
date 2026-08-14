# benchmarking/reset_results.py

"""Removes the 'leanexplore_results' key from JSON objects in a file.

This script reads a JSON file, which is expected to contain either a list of
JSON objects or a single JSON object. It iterates through the objects (or
processes the single object) and removes the 'leanexplore_results' key if
it exists. The modified data is then written back to the same file.

This is typically used to reset or clean up search results for a new
benchmarking study.
"""

import json


def remove_leanexplore_results_from_list(filepath: str) -> None:
    """Resets the results for next study.

    Reads a JSON file containing a list of objects,
    removes the 'leanexplore_results' key from each object in the list,
    and writes back to the file.

    Args:
        filepath: The path to the JSON file.
    """
    try:
        with open(filepath) as f:
            data_list = json.load(f)

        if not isinstance(data_list, list):
            print(
                f"Error: Expected a list of JSON objects in {filepath},"
                f" but found {type(data_list)}."
            )
            if isinstance(data_list, dict) and "leanexplore_results" in data_list:
                del data_list["leanexplore_results"]
                print(
                    "Processed as a single object: Successfully removed"
                    f" 'leanexplore_results' from {filepath}"
                )
                with open(filepath, "w") as f:
                    json.dump(data_list, f, indent=4)
            else:
                print(
                    "Could not process the file as a list or a single object with the"
                    " target key."
                )
            return

        modified_count = 0
        for item in data_list:
            if isinstance(item, dict) and "leanexplore_results" in item:
                del item["leanexplore_results"]
                modified_count += 1

        if modified_count > 0:
            print(
                f"Successfully removed 'leanexplore_results' from {modified_count}"
                f" object(s) in {filepath}"
            )
        else:
            print(
                f"'leanexplore_results' key not found in any of "
                f"the objects in {filepath}"
            )

        with open(filepath, "w") as f:
            json.dump(data_list, f, indent=4)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from {filepath}. Ensure it's a valid JSON"
            " array."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    file_path = "benchmarking/search_results.json"
    remove_leanexplore_results_from_list(file_path)
