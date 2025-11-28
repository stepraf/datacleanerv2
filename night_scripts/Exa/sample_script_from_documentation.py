import exa_py

from config import (
    EXA_API_KEY
)
print("Hello!")

exa = exa_py.Exa(EXA_API_KEY)

queries = ["Who is the manufacturer of the product BGW620, which companies use it, what is the main SOC of the product?"]

def get_search_results(queries, links_per_query=2):
    results = []
    print("Getting search results for queries:")
    print(queries)
    for query in queries:
        print("Query:")
        print(query)
        search_response = exa.answer(query,
            num_results=links_per_query
        )
        results.extend(search_response.results)
        print("Results:")
        print(results)
    return results

get_search_results(queries)


