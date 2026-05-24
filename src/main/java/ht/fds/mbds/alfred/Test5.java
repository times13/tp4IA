package ht.fds.mbds.alfred;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import java.util.List;
import java.util.Scanner;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import java.util.Map;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;

public class Test5 {

    public static void main(String[] args) {

        String geminiKey =
                System.getenv("GEMINI_KEY");

        String tavilyKey =
                System.getenv("TAVILY_KEY");

        ChatModel model =
                GoogleAiGeminiChatModel.builder()
                        .apiKey(geminiKey)
                        .modelName("gemini-2.5-flash-lite")
                        .temperature(0.2)
                        .logRequestsAndResponses(true)
                        .build();

        EmbeddingStore<TextSegment> store =
                creerEmbeddingStore(
                        "rag.pdf"
                );

        ContentRetriever pdfRetriever =
                EmbeddingStoreContentRetriever
                        .builder()
                        .embeddingStore(store)
                        .embeddingModel(
                                new AllMiniLmL6V2EmbeddingModel()
                        )
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        WebSearchEngine webSearchEngine =
                TavilyWebSearchEngine
                        .builder()
                        .apiKey(tavilyKey)
                        .build();

        ContentRetriever webRetriever =
                WebSearchContentRetriever
                        .builder()
                        .webSearchEngine(
                                webSearchEngine
                        )
                        .build();
/*
        DefaultQueryRouter queryRouter =
                new DefaultQueryRouter(
                        pdfRetriever,
                        webRetriever
                );
*/
// ===== LanguageModelQueryRouter =====
        // On associe chaque retriever à une description en langage naturel.
        // Le LLM utilise ces descriptions pour décider quel(s) retriever(s)
        // appeler en fonction de la question posée.

        Map<ContentRetriever, String> retrieverDescriptions = Map.of(
                pdfRetriever,
                "Documents sur l'intelligence artificielle, le RAG "
                        + "(Retrieval Augmented Generation) et le fine-tuning",
                webRetriever,
                "Informations générales et actualités disponibles sur le web"
        );

        // Constructeur direct : (ChatModel, Map<ContentRetriever, String>)
        LanguageModelQueryRouter queryRouter =
                new LanguageModelQueryRouter(model, retrieverDescriptions);

        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(model)
                        .chatMemory(
                                MessageWindowChatMemory.withMaxMessages(10)
                        )
                        .retrievalAugmentor(retrievalAugmentor)
                        .build();

        conversationAvec(assistant);
    }



    private static EmbeddingStore<TextSegment>
    creerEmbeddingStore(String nomFichier){

        Document document =
                ClassPathDocumentLoader.loadDocument(
                        nomFichier,
                        new ApacheTikaDocumentParser()
                );

        List<TextSegment> segments =
                DocumentSplitters
                        .recursive(300,30)
                        .split(document);

        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        List<Embedding> embeddings =
                embeddingModel
                        .embedAll(segments)
                        .content();

        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        store.addAll(
                embeddings,
                segments
        );

        return store;
    }



    private static void conversationAvec(
            Assistant assistant){

        Scanner scanner =
                new Scanner(System.in);

        while(true){

            System.out.println(
                    "\nQuestion : "
            );

            String question =
                    scanner.nextLine();

            if(question.equalsIgnoreCase(
                    "fin"
            )){
                break;
            }

            System.out.println(
                    assistant.chat(question)
            );

        }

    }

}