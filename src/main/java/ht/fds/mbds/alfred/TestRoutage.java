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

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.*;

public class TestRoutage {

    public static void main(String[] args) {

        String apiKey = System.getenv("GEMINI_KEY");

        ChatModel model =
                GoogleAiGeminiChatModel.builder()
                        .apiKey(apiKey)
                        .modelName("gemini-2.5-flash")
                        .temperature(0.2)
                        .logRequestsAndResponses(true)
                        .build();

        // ===== PHASE 1 : ingestion =====

        EmbeddingStore<TextSegment> storeIA =
                creerEmbeddingStore("rag.pdf");

        EmbeddingStore<TextSegment> storeAutre =
                creerEmbeddingStore("cours_intro_reseaux.pdf");

        // ===== PHASE 2 =====

        ContentRetriever retrieverIA =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(storeIA)
                        .embeddingModel(
                                new AllMiniLmL6V2EmbeddingModel()
                        )
                        .maxResults(2)
                        .minScore(0.5)
                        .build();


        ContentRetriever retrieverAutre =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(storeAutre)
                        .embeddingModel(
                                new AllMiniLmL6V2EmbeddingModel()
                        )
                        .maxResults(2)
                        .minScore(0.5)
                        .build();



        Map<ContentRetriever,String> descriptions =
                new HashMap<>();

        descriptions.put(
                retrieverIA,
                "Support de cours sur l'intelligence artificielle, RAG et fine-tuning"
        );

        descriptions.put(
                retrieverAutre,
                "Support de cours sur les réseaux informatiques, protocoles réseau, services réseau et architecture des réseaux"
        );


        QueryRouter queryRouter =
                new LanguageModelQueryRouter(
                        model,
                        descriptions
                );


        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();


        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(model)
                        .chatMemory(
                                MessageWindowChatMemory
                                        .withMaxMessages(10)
                        )
                        .retrievalAugmentor(
                                retrievalAugmentor
                        )
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

            System.out.println("\nQuestion : ");

            String question =
                    scanner.nextLine();

            if(question.equalsIgnoreCase("fin")){
                break;
            }

            System.out.println(
                    assistant.chat(question)
            );
        }

    }

}