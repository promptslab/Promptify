def ner(text_input, config = None):
    
    main_config       = {'task'          :  'ner',
                         'description'   :  '',
                         'sub_task'      :  '', 
                         'domain'        :  '',
                         'example'       :  '',
                         'n_shots'       :  1,
                         'n_ner'         :  10,
                         'output_format' :  {"entity_group":"",
                                                "score":"",
                                                "word":"",
                                                "start":"",
                                                "end":""
                                             }}

        
    
    if config:
        main_config.update(config)
    
    
    shots_template = "Below is an example of Named Entity Recognition task \n\n" + str(get_shots(main_config['n_shots'], 
                                                                                               main_config['domain']))
    if main_config['n_ner']!='':
        template = shots_template + "\n\nPerform "  + str(main_config['n_ner']) + " Named Entity Recognition on the following paragraph, the format should be as described in the above example\n\n" + text_input
        return template
    else:
        template = shots_template + "\n\nPerform Named Entity Recognition on the below paragraph, the format should be as described in the above example\n\n" + text_input
        return template
