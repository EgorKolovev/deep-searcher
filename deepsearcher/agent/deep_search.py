import asyncio
from typing import List, Tuple

from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.utils import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results

SUB_QUERY_PROMPT = """Чтобы ответить на этот вопрос более подробно, разбейте исходный вопрос на четыре подвопроса. Верните в виде списка str.
Если это очень простой вопрос и разложение не требуется, то оставьте единственный исходный вопрос в виде списка Python.

Исходный вопрос: {original_query}

<EXAMPLE>
Пример ввода:
"Объясните глубокое обучение"

Пример вывода:
[
"Что такое глубокое обучение?",
"В чем разница между глубоким обучением и машинным обучением?",
"Какова история глубокого обучения?"
]
</EXAMPLE>

Предоставьте свой ответ в виде списка Python в формате str:
"""

RERANK_PROMPT = """На основе вопросов запроса и извлеченного фрагмента, чтобы определить, полезен ли фрагмент для ответа на любой из вопросов запроса, вы можете вернуть только "YES" или "NO" без какой-либо другой информации.

Вопросы запроса: {query}
Извлеченный фрагмент: {retrieved_chunk}

Помог ли фрагмент для ответа на любой из вопросов?
"""


REFLECT_PROMPT = """Определите, нужны ли дополнительные поисковые запросы, на основе исходного запроса, предыдущих подзапросов и всех извлеченных фрагментов документа. Если требуется дополнительное исследование, предоставьте список Python из 3 поисковых запросов. Если дополнительное исследование не требуется, верните пустой список.

Если исходный запрос заключается в написании отчета, то вы предпочитаете сгенерировать несколько дополнительных запросов, а не возвращать пустой список.

Исходный запрос: {question}

Предыдущие подзапросы: {mini_questions}

Связанные фрагменты:
{mini_chunk_str}

Отвечайте исключительно в допустимом формате списка str(list of str) без какого-либо другого текста."""


SUMMARY_PROMPT = """Вы эксперт по анализу контента с использованием искусственного интеллекта, хорошо умеете резюмировать контент. Пожалуйста, резюмируйте конкретный и подробный ответ или отчет на основе предыдущих запросов и извлеченных фрагментов документа.

Исходный запрос: {question}

Предыдущие подзапросы: {mini_questions}

Связанные фрагменты:
{mini_chunk_str}

"""


@describe_class(
    "This agent is suitable for handling general and simple queries, such as given a topic and then writing a report, survey, or article."
)
class DeepSearch(RAGAgent):
    """
    Deep Search agent implementation for comprehensive information retrieval.

    This agent performs a thorough search through the knowledge base, analyzing
    multiple aspects of the query to provide comprehensive and detailed answers.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        max_iter: int = 3,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        **kwargs,
    ):
        """
        Initialize the DeepSearch agent.

        Args:
            llm: The language model to use for generating answers.
            embedding_model: The embedding model to use for query embedding.
            vector_db: The vector database to search for relevant documents.
            max_iter: The maximum number of iterations for the search process.
            route_collection: Whether to use a collection router for search.
            text_window_splitter: Whether to use text_window splitter.
            **kwargs: Additional keyword arguments for customization.
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.max_iter = max_iter
        self.route_collection = route_collection
        self.collection_router = CollectionRouter(
            llm=self.llm, vector_db=self.vector_db, dim=embedding_model.dimension
        )
        self.text_window_splitter = text_window_splitter

    def _generate_sub_queries(self, original_query: str) -> Tuple[List[str], int]:
        chat_response = self.llm.chat(
            messages=[
                {"role": "user", "content": SUB_QUERY_PROMPT.format(original_query=original_query)}
            ]
        )
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    async def _search_chunks_from_vectordb(self, query: str, sub_queries: List[str]):
        consume_tokens = 0
        if self.route_collection:
            selected_collections, n_token_route = self.collection_router.invoke(
                query=query, dim=self.embedding_model.dimension
            )
        else:
            selected_collections = self.collection_router.all_collections
            n_token_route = 0
        consume_tokens += n_token_route

        all_retrieved_results = []
        query_vector = self.embedding_model.embed_query(query)
        for collection in selected_collections:
            log.color_print(f"<search> Search [{query}] in [{collection}]...  </search>\n")
            retrieved_results = self.vector_db.search_data(
                collection=collection, vector=query_vector
            )
            if not retrieved_results or len(retrieved_results) == 0:
                log.color_print(
                    f"<search> No relevant document chunks found in '{collection}'! </search>\n"
                )
                continue
            accepted_chunk_num = 0
            references = set()
            for retrieved_result in retrieved_results:
                chat_response = self.llm.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": RERANK_PROMPT.format(
                                query=[query] + sub_queries,
                                retrieved_chunk=f"<chunk>{retrieved_result.text}</chunk>",
                            ),
                        }
                    ]
                )
                consume_tokens += chat_response.total_tokens
                response_content = chat_response.content.strip()
                # strip the reasoning text if exists
                if "<think>" in response_content and "</think>" in response_content:
                    end_of_think = response_content.find("</think>") + len("</think>")
                    response_content = response_content[end_of_think:].strip()
                if "YES" in response_content and "NO" not in response_content:
                    all_retrieved_results.append(retrieved_result)
                    accepted_chunk_num += 1
                    references.add(retrieved_result.reference)
            if accepted_chunk_num > 0:
                log.color_print(
                    f"<search> Accept {accepted_chunk_num} document chunk(s) from references: {list(references)} </search>\n"
                )
            else:
                log.color_print(
                    f"<search> No document chunk accepted from '{collection}'! </search>\n"
                )
        return all_retrieved_results, consume_tokens

    def _generate_gap_queries(
        self, original_query: str, all_sub_queries: List[str], all_chunks: List[RetrievalResult]
    ) -> Tuple[List[str], int]:
        reflect_prompt = REFLECT_PROMPT.format(
            question=original_query,
            mini_questions=all_sub_queries,
            mini_chunk_str=self._format_chunk_texts([chunk.text for chunk in all_chunks])
            if len(all_chunks) > 0
            else "NO RELATED CHUNKS FOUND.",
        )
        chat_response = self.llm.chat([{"role": "user", "content": reflect_prompt}])
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    def retrieve(self, original_query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Retrieve relevant documents from the knowledge base for the given query.

        This method performs a deep search through the vector database to find
        the most relevant documents for answering the query.

        Args:
            original_query (str): The query to search for.
            **kwargs: Additional keyword arguments for customizing the retrieval.

        Returns:
            Tuple[List[RetrievalResult], int, dict]: A tuple containing:
                - A list of retrieved document results
                - The token usage for the retrieval operation
                - Additional information about the retrieval process
        """
        return asyncio.run(self.async_retrieve(original_query, **kwargs))

    async def async_retrieve(
        self, original_query: str, **kwargs
    ) -> Tuple[List[RetrievalResult], int, dict]:
        max_iter = kwargs.pop("max_iter", self.max_iter)
        ### SUB QUERIES ###
        log.color_print(f"<query> {original_query} </query>\n")
        all_search_res = []
        all_sub_queries = []
        total_tokens = 0

        sub_queries, used_token = self._generate_sub_queries(original_query)
        total_tokens += used_token
        if not sub_queries:
            log.color_print("No sub queries were generated by the LLM. Exiting.")
            return [], total_tokens, {}
        else:
            log.color_print(
                f"<think> Break down the original query into new sub queries: {sub_queries}</think>\n"
            )
        all_sub_queries.extend(sub_queries)
        sub_gap_queries = sub_queries

        for iter in range(max_iter):
            log.color_print(f">> Iteration: {iter + 1}\n")
            search_res_from_vectordb = []
            search_res_from_internet = []  # TODO

            # Create all search tasks
            search_tasks = [
                self._search_chunks_from_vectordb(query, sub_gap_queries)
                for query in sub_gap_queries
            ]
            # Execute all tasks in parallel and wait for results
            search_results = await asyncio.gather(*search_tasks)
            # Merge all results
            for result in search_results:
                search_res, consumed_token = result
                total_tokens += consumed_token
                search_res_from_vectordb.extend(search_res)

            search_res_from_vectordb = deduplicate_results(search_res_from_vectordb)
            # search_res_from_internet = deduplicate_results(search_res_from_internet)
            all_search_res.extend(search_res_from_vectordb + search_res_from_internet)
            if iter == max_iter - 1:
                log.color_print("<think> Exceeded maximum iterations. Exiting. </think>\n")
                break
            ### REFLECTION & GET GAP QUERIES ###
            log.color_print("<think> Reflecting on the search results... </think>\n")
            sub_gap_queries, consumed_token = self._generate_gap_queries(
                original_query, all_sub_queries, all_search_res
            )
            total_tokens += consumed_token
            if not sub_gap_queries or len(sub_gap_queries) == 0:
                log.color_print("<think> No new search queries were generated. Exiting. </think>\n")
                break
            else:
                log.color_print(
                    f"<think> New search queries for next iteration: {sub_gap_queries} </think>\n"
                )
                all_sub_queries.extend(sub_gap_queries)

        all_search_res = deduplicate_results(all_search_res)
        additional_info = {"all_sub_queries": all_sub_queries}
        return all_search_res, total_tokens, additional_info

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Query the agent and generate an answer based on retrieved documents.

        This method retrieves relevant documents and uses the language model
        to generate a comprehensive answer to the query.

        Args:
            query (str): The query to answer.
            **kwargs: Additional keyword arguments for customizing the query process.

        Returns:
            Tuple[str, List[RetrievalResult], int]: A tuple containing:
                - The generated answer
                - A list of retrieved document results
                - The total token usage
        """
        all_retrieved_results, n_token_retrieval, additional_info = self.retrieve(query, **kwargs)
        if not all_retrieved_results or len(all_retrieved_results) == 0:
            return f"No relevant information found for query '{query}'.", [], n_token_retrieval
        all_sub_queries = additional_info["all_sub_queries"]
        chunk_texts = []
        for chunk in all_retrieved_results:
            if self.text_window_splitter and "wider_text" in chunk.metadata:
                chunk_texts.append(chunk.metadata["wider_text"])
            else:
                chunk_texts.append(chunk.text)
        log.color_print(
            f"<think> Summarize answer from all {len(all_retrieved_results)} retrieved chunks... </think>\n"
        )
        summary_prompt = SUMMARY_PROMPT.format(
            question=query,
            mini_questions=all_sub_queries,
            mini_chunk_str=self._format_chunk_texts(chunk_texts),
        )
        chat_response = self.llm.chat([{"role": "user", "content": summary_prompt}])
        log.color_print("\n==== FINAL ANSWER====\n")
        log.color_print(chat_response.content)
        return (
            chat_response.content,
            all_retrieved_results,
            n_token_retrieval + chat_response.total_tokens,
        )

    def _format_chunk_texts(self, chunk_texts: List[str]) -> str:
        chunk_str = ""
        for i, chunk in enumerate(chunk_texts):
            chunk_str += f"""<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"""
        return chunk_str
