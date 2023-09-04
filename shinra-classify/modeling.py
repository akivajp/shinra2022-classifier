import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer.backends import cuda
from sklearn.metrics import precision_recall_fscore_support


initializer = chainer.initializers


class ENEClassifier(chainer.Chain):
    def __init__(self, feature_vocab_size, feature_embed_size,
                 entity_vocab_size, entity_embed_size,
                 hidden_size, out_size, dropout=0.0,
                 feature_initial_embed=None, entity_initial_embed=None):
        super(ENEClassifier, self).__init__()
        with self.init_scope():
            self.embed_feature = L.EmbedID(feature_vocab_size, feature_embed_size,
                initialW=feature_initial_embed, ignore_label=-1)
            self.embed_entity = L.EmbedID(entity_vocab_size, entity_embed_size,
                initialW=entity_initial_embed, ignore_label=-1)
            self.encoder = L.Linear(feature_embed_size + entity_embed_size, hidden_size,
                initialW=I.Uniform(0.1))
            self.decoder = L.Linear(hidden_size, out_size, initialW=I.Uniform(0.1))

        self.dropout = dropout

    def compute_logits(self, feature_ids, entity_id):
        feature_embed = F.sum(self.embed_feature(feature_ids), axis=1)
        entity_embed = self.embed_entity(entity_id)
        concat_embed = F.concat((feature_embed, entity_embed), axis=1)
        concat_embed = F.dropout(concat_embed, self.dropout)
        hidden = self.encoder(concat_embed)
        hidden = F.tanh(hidden)
        hidden = F.dropout(hidden, self.dropout)
        logits = self.decoder(hidden)

        return logits

    def forward(self, feature_ids, entity_id, label_ids):
        logits = self.compute_logits(feature_ids, entity_id)
        loss = F.sigmoid_cross_entropy(logits, label_ids)
        chainer.reporter.report({'loss': loss}, self)

        return loss

    #def predict(self, feature_ids, entity_id, label_ids=None):
    def predict(self, feature_ids, entity_id, label_ids=None, nbest=1):
        logits = self.compute_logits(feature_ids, entity_id)
        probs = F.sigmoid(logits)
        probs = cuda.to_cpu(probs.data)
        xp = self.xp
        #pred_ids = ((probs > 0.5) + (probs == probs.max(axis=1)[:, None])).astype('i')
        pred_ids = ((probs > 0.5) + (probs >= xp.sort(probs, axis=1)[:, -nbest][:, None])).astype('i')
        if label_ids is not None:
            loss = F.sigmoid_cross_entropy(logits, label_ids)
            chainer.reporter.report({'loss': loss}, self)

            label_ids = cuda.to_cpu(label_ids)
            (precision, recall, f1_score, _) = \
                precision_recall_fscore_support(label_ids, pred_ids, average='samples')
            chainer.reporter.report({'precision': precision}, self)
            chainer.reporter.report({'recall': recall}, self)
            chainer.reporter.report({'f1_score': f1_score}, self)

        return pred_ids, probs
