import json
from IPython.core.display import display, HTML, Javascript
import os
from .util import format_special_chars, format_attention

def model_view(attention, tokens, sentence_b_start=None, prettify_tokens=True):
    """Render model view

        Args:
            attention: list of ``torch.FloatTensor``(one for each layer) of shape
                ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
            tokens: list of tokens
            sentence_b_index: index of first wordpiece in sentence B if input text is sentence pair (optional)
            prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
    """

    if sentence_b_start is not None:
        vis_html = """
        <!DOCTYPE html>
        <html lang="en">
        <script src="https://d3js.org/d3.v5.min.js"></script>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.2/jquery.min.js"></script>
        <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"></script>
        <body> 
        <span style="user-select:none">
            Attention: <select id="filter">
              <option value="all">All</option>
              <option value="aa">Sentence A -> Sentence A</option>
              <option value="ab">Sentence A -> Sentence B</option>
              <option value="ba">Sentence B -> Sentence A</option>
              <option value="bb">Sentence B -> Sentence B</option>
            </select>
            </span>
        <div id='vis'></div>
        <script type="text/javascript">
        """
    else:
        vis_html = """
        <!DOCTYPE html>
        <html lang="en">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min.js"></script>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.2/jquery.min.js"></script>
        <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"></script>
        <body>
          <div id='vis'></div> 
          <script type="text/javascript">
        """

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), 'bertviz'))
    vis_js = open(os.path.join(__location__, 'model_view.js')).read()

    if prettify_tokens:
        tokens = format_special_chars(tokens)

    attn = format_attention(attention)
    attn_data = {
        'all': {
            'attn': attn.tolist(),
            'left_text': tokens,
            'right_text': tokens
        }
    }
    if sentence_b_start is not None:
        slice_a = slice(0, sentence_b_start)  # Positions corresponding to sentence A in input
        slice_b = slice(sentence_b_start, len(tokens))  # Position corresponding to sentence B in input
        attn_data['aa'] = {
            'attn': attn[:, :, slice_a, slice_a].tolist(),
            'left_text': tokens[slice_a],
            'right_text': tokens[slice_a]
        }
        attn_data['bb'] = {
            'attn': attn[:, :, slice_b, slice_b].tolist(),
            'left_text': tokens[slice_b],
            'right_text': tokens[slice_b]
        }
        attn_data['ab'] = {
            'attn': attn[:, :, slice_a, slice_b].tolist(),
            'left_text': tokens[slice_a],
            'right_text': tokens[slice_b]
        }
        attn_data['ba'] = {
            'attn': attn[:, :, slice_b, slice_a].tolist(),
            'left_text': tokens[slice_b],
            'right_text': tokens[slice_a]
        }
    params = {
        'attention': attn_data,
        'default_filter': "all"
    }
    attn_seq_len = len(attn_data['all']['attn'][0][0])
    if attn_seq_len != len(tokens):
        raise ValueError(f"Attention has {attn_seq_len} positions, while number of tokens is {len(tokens)}")
    import json
    vis_html=vis_html+'\nvar curparams=' +json.dumps(params)+'\n;'+ vis_js+ '</script> </body> </html>'

    with open("a13.html", "w") as text_file:
        text_file.write(vis_html)
