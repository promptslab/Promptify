def Summary(
        context: str,  # the context paragraph
        domain: str = "",  # the domain of the question, optional
        token_length= ""
          ):
        """
        This function returns a template string with the input context. The template also includes a description of the function's purpose. If a domain is provided, it is
        included in the function's description.
        """

        if domain:
            if token_length!="":
                template = f"You are a highly intelligent {domain} domain expert Summarization system. You take Passage as input and summarize the passage as a {domain} expert in {token_length} tokens. Your output format is only [{{'S': Summarization paragraph}},] form, no other form.\n"
            else:
                template = f"You are a highly intelligent {domain} domain expert Summarization system. You take Passage as input and summarize the passage as a {domain} expert. Your output format is only [{{'S': Summarization paragraph}},] form, no other form.\n"
        
        else:
            template = "You are a highly intelligent Summarization system. You take Passage as input and summarize the passage as an expert. Your output format is only [{'S': Summarization paragraph},] form, no other form.\n"

        # add the input context and question to the template
        template += f"\nPassage: {context}\nOutput:"
        return template
