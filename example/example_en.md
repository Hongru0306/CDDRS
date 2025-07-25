# GKGR Test Case: A Christmas Carol

We use [A Christmas Carol](https://raw.githubusercontent.com/Hongru0306/CDDRS/example/english-text.txt) as the test corpus.

---

## Sample Question

**Question:**  
What did the woman who took the bed-curtains from the dead man say about putting the shirt on the body for burial?

---

## Retrieval and Generation Comparison

### 1. Ordinary RAG Retrieval

**Retrieved Fragment:**

The room was very dark, too dark to be observed with any accuracy, though Scrooge glanced round it in obedience to a secret impulse, anxious to know what kind of room it was. A pale light, rising in the outer air, fell straight upon the bed; and on it, plundered and bereft, unwatched, unwept, uncared for, was the body of this man.

**Generated Answer:**

The woman entered the dark room where the dead man lay, and took items from his surroundings. The body was left alone, uncared for, and the atmosphere was cold and stripped of comfort. The text describes the scene in the room but does not detail the woman's comments about the burial shirt.

---

### 2. GKGR RAG Retrieval

**Retrieved Fragment:**

'You don't mean to say you took 'em down, rings and all, with him lying there?' said Joe.

'Yes, I do,' replied the woman. 'Why not?'

'Ah! you may look through that shirt till your eyes ache, but you won't find a hole in it, nor a threadbare place. It's the best he had, and a fine one too. They'd have wasted it, if it hadn't been for me.'

'What do you call wasting of it?' asked old Joe.

'Putting it on him to be buried in, to be sure,' replied the woman, with a laugh. 'Somebody was fool enough to do it, but I took it off again. If calico an't good enough for such a purpose, it isn't good enough for anything. It's quite as becoming to the body. He can't look uglier than he did in that one.'

**Generated Answer:**

The woman who took the bed-curtains from the dead man said that she removed his shirt because it would have been a waste to bury him in such a fine one. She said, "Ah! you may look through that shirt till your eyes ache, but you won't find a hole in it, nor a threadbare place. It's the best he had, and a fine one too. They'd have wasted it, if it hadn't been for me." When asked what she meant by wasting it, she replied, "Putting it on him to be buried in, to be sure. Somebody was fool enough to do it, but I took it off again. If calico an't good enough for such a purpose, it isn't good enough for anything. It's quite as becoming to the body. He can't look uglier than he did in that one."

---
