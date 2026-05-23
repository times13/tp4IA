package ht.fds.mbds.alfred;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import java.util.List;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagNaif {

    public static void main(String[] args) {

        configureLogger();
        // ===== Chargement du PDF =====

        Document document =
                ClassPathDocumentLoader.loadDocument(
                        "rag.pdf",
                        new ApacheTikaDocumentParser()
                );

        // ===== Découpage du document =====

        List<TextSegment> segments =
                DocumentSplitters
                        .recursive(300,30)
                        .split(document);

        // ===== Création du modèle d'embeddings =====

        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        // ===== Création des embeddings =====

        List<Embedding> embeddings =
                embeddingModel
                        .embedAll(segments)
                        .content();

        // ===== Base vectorielle mémoire =====

        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        // ===== Association embeddings + segments =====

        embeddingStore.addAll(
                embeddings,
                segments
        );

        System.out.println(
                "Embeddings enregistrés : "
                        + embeddings.size()
        );

        String llmKey = System.getenv("GEMINI_KEY");

        if(llmKey == null){
            System.out.println("Variable GEMINI_KEY absente");
            return;
        }

        ChatModel model =
                GoogleAiGeminiChatModel.builder()
                        .apiKey(llmKey)
                        .modelName("gemini-2.5-flash")
                        .temperature(0.2)
                        .logRequestsAndResponses(true)
                        .build();

        EmbeddingStoreContentRetriever retriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(model)
                        .chatMemory(
                                MessageWindowChatMemory
                                        .withMaxMessages(10)
                        )
                        .contentRetriever(retriever)
                        .build();

        System.out.println(
                assistant.chat(
                        "Quelle est la signification de RAG ; à quoi ça sert ?"
                )
        );

        conversationAvec(assistant);
    }

    private static void conversationAvec(
            Assistant assistant){

        try (Scanner scanner =
                     new Scanner(System.in)) {

            while(true){

                System.out.println(
                        "================================"
                );

                System.out.println(
                        "Posez votre question : "
                );

                String question =
                        scanner.nextLine();

                if(question.isBlank()){
                    continue;
                }

                if("fin".equalsIgnoreCase(question)){
                    break;
                }

                String reponse =
                        assistant.chat(question);

                System.out.println(
                        "Assistant : "
                                + reponse
                );
            }
        }
    }

    private static void configureLogger() {

        Logger packageLogger =
                Logger.getLogger("dev.langchain4j");

        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler =
                new ConsoleHandler();

        handler.setLevel(Level.FINE);

        packageLogger.addHandler(handler);
    }
}

