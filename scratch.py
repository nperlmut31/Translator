import torch
from nets.seq_to_seq_model import Seq2Seq_Att
from nets.teacher_forcing import TeacherForcing
from nets.beam_search import BeamSearch

with open('test_batch.pt', 'rb') as f:
    batch = torch.load(f)
    f.close()

input, target = batch[0], batch[1]

model = Seq2Seq_Att(embedding_dim=4, encoder_state_size=4)

trainer = TeacherForcing(model = model)
beam_search = BeamSearch(model=model, beam_size=2)

score = beam_search(input)

print(score)