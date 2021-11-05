from numpy import average
from SPARQLWrapper import SPARQLWrapper
import argparse
import pickle
import time


class QueryTester(object):
    """ Object for testing phase """

    def __init__(self, test_queries: str, query_output: str, golden_standard: str = None) -> None:
        self.test_queries = test_queries
        self.query_output = query_output
        self.golden_standard = golden_standard
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.sparql.setReturnFormat("json")

    def resolve_queries(self) -> None:
        """Requests the queries in test_queries and saves the results"""
        input_file = open(self.test_queries, 'r', encoding='UTF-8')
        output_file = open(self.query_output, 'w+', encoding='UTF-8')

        pickle_list = []
        for query in input_file.readlines():
            self.sparql.setQuery(query)
            try:
                ret = self.sparql.query().convert()
                answer_list = ret["results"]["bindings"]
                answer_uris = set()
                for answer in answer_list:
                    answer_uris.add(answer["uri"]["value"])
                output_file.write(str(answer_uris) + '\n')
                pickle_list.append(answer_uris)
                # ret is a stream with the results in XML, see <http://www.w3.org/TR/rdf-sparql-XMLres/>
            except Exception as e:
                print(f"ERROR WITH QUERY: {e}")
            time.sleep(1)
        pickle.dump(pickle_list, open(f"{self.query_output[:-3]}p", "wb"))

    def compare_queries(self):
        if self.test_queries:
            self.resolve_queries()
            query_outputs = pickle.load(open(f"{self.query_output[:-3]}p", "rb"))
        else:
            query_outputs = pickle.load(open(self.query_output, "rb"))
        golden_standard = pickle.load(open(self.golden_standard, "rb"))

        # normal accuracy
        correct = 0
        for i in range(len(query_outputs)):
            y_pred = query_outputs[i]
            y_true = golden_standard[i]
            intersect = y_pred.intersection(y_true)
            precision = len(intersect) / len(y_pred)
            recall = len(intersect) / len(y_true)
            try:
                f1_score = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1_score = 0
            print(f1_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_queries", dest="test_queries")
    parser.add_argument("--query_output", dest="query_output", required=True)
    parser.add_argument("--golden_standard", dest="golden_standard")
    args = parser.parse_args()
    obj = QueryTester(args.test_queries, args.query_output, args.golden_standard)
    if obj.golden_standard:
        obj.compare_queries()
    else:
        obj.resolve_queries()


if __name__ == '__main__':
    main()


