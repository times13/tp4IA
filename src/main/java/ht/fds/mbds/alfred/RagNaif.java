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

public class RagNaif {

    public static void main(String[] args) {

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
    }
}