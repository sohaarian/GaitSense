# Research narrative (draft for the personal statement)

*This is a draft to rewrite in your own words. Keep the facts and change the voice.*

**One-line description for the application:** Testing whether affordable sensing can detect meaningful change in injured workers' walking. Across three studies of 1,667 workers' compensation rehabilitation patients, I showed that standard change limits mistake ordinary variation for change and built a corrected limit that held across four injury groups. I then found that AI recovers some push-off information from vertical force alone, about a third of real changes, but not enough to replace force plates. That result sets the requirements for a future wearable.

---

Working in insurance claims, I saw how often decisions about an injured worker's recovery rest on narrative notes and single snapshots. Objective measurements exist, but they usually come from expensive laboratory equipment that most clinics do not have, and nobody checks whether a change in them means what people assume. I wanted to know whether affordable technology could detect real changes in how an injured person walks.

I used GaitRec, an open dataset of 1,667 patients with knee, ankle, hip and heel injuries from a rehabilitation centre run by the Austrian Workers' Compensation Board.

My first finding was about trust. The usual way of deciding whether a change between visits is real flagged 28% of changes when it should have flagged 5%. Measurements simply move more between visits than within one. I built a corrected limit, tested it on patients it was not fitted to, and showed that one rule works for all four injury groups.

Next I asked whether cheap sensing could do the laboratory's job. A pressure insole measures only vertical force, while push-off strength, which matters for recovery, needs a force plate. I trained models, including a convolutional network written from scratch and verified numerically, to recover push-off from the vertical signal alone. The shape of the vertical curve turned out to hold real information that its peak does not. When push-off changed, the model usually got the direction right. But it caught only about a third of real changes.

Finally, under an analysis plan I froze in advance, I tested whether detecting change directly, rather than reconstructing it, would do better. It did not. I also tested this on later patients I had kept aside. The honest conclusion was that, with the models I tested, vertical force alone cannot reliably detect most of these changes. The next step is a sensor that measures push-off directly.

That changed how I think about medical AI. A model can look accurate and still fail at the job it is meant for, and sometimes the right answer is a better measurement, not a bigger network. I learned to define the clinical question first, freeze my plan before looking at results, test on patients the model has never seen, and report a negative result when that is what the data show.

At Yale, I want to learn how to take this from retrospective data to a device study with real patients and clinical outcomes: the sensor validation, the clinical meaning of a change, and the regulatory and human-factors work that medical software needs. I am especially interested in [Analyzing Medical Data II and AI Software Engineering in a Regulated Context].
