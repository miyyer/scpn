# scpn

Code to train models from "Adversarial Example Generation with Syntactically Controlled Paraphrase Networks".

The code is written in python and requires Pytorch 3.1.

To get started, download trained models (scpn.pt and parse_generator.pt) from https://drive.google.com/file/d/1AuH1aHrE9maYttuSJz_9eltYOAad8Mfj/view?usp=sharing and place them in the models directory.

To train, download the training data from https://drive.google.com/file/d/1x1Xz3KQP_Ncu3DVPhOCsAwwOlFgcre5H/view?usp=sharing and move it to the data folder. Check train_scpn.py for training command line options. To train a model from scratch with default settings run train.sh.

There is also a demo script (run demo.sh) that will generate paraphrases from a set of templates (check the script to see available choices).

If you use our code or models for your work please cite:

@inproceedings{iyyer-2018-controlled, author = {Iyyer, Mohit and Wieting, John and Gimpel, Kevin and Zettlemoyer, Luke}, title = {Adversarial Example Generation with Syntactically Controlled Paraphrase Networks}, booktitle={Proceedings of NAACL}, year = {2018} }

If you use the data please cite the above and:

@inproceedings{wieting-17-millions, author = {John Wieting and Kevin Gimpel}, title = {Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations}, booktitle = {Proceedings of ACL}, year = {2018} }