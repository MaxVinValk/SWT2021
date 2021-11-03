from SPARQLWrapper import SPARQLWrapper
import time

class Query_tester(object):
    """ Object for testing phase """

    def __init__(self, query_output: str, test_output: str = None, golden_standard: str = None) -> None:
        self.test_output = test_output
        self.golden_standard = golden_standard
        self.query_output = query_output
        self.sparql = SPARQLWrapper("https://query.wikidata.org/")

    def resolve_queries(self) -> None:
        """Requests the queries in test_output"""
        input_file = open(self.test_output, 'r')
        output_file = open(self.query_output, 'w+')
        for query in input_file.readlines():
            self.sparql.setQuery(query)
            try:
                ret = self.sparql.query()
                print(ret)
                # ret is a stream with the results in XML, see <http://www.w3.org/TR/rdf-sparql-XMLres/>
            except Exception as e:
                print(f"ERROR WITH QUERY: {e}")
                ret = f"ERROR WITH QUERY: {e}"
            time.sleep(20)

    def compare_queries(self):
        self.resolve_queries()
        query_outputs = open(self.query_output, 'r').readlines()
        golden_standard = open(self.golden_standard, 'r').readlines()

        # normal accuracy
        correct = 0
        for i in range(len(query_outputs)):
            if query_outputs[i] == golden_standard[i]:
                correct += 1
        accuracy = correct / len(query_outputs)

obj = Query_tester("data/results.txt", "data/input.txt", "data/output.txt")
obj.resolve_queries()


