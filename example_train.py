# Prepare training data loader
        train_examples = read_examples(
            args.train_filename + "." + args.source,
            args.train_filename + "." + args.target,
        )
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args, stage="train"
        )
        all_source_ids = torch.tensor(
            [f.source_ids for f in train_features], dtype=torch.long
        )
        all_source_mask = torch.tensor(
            [f.source_mask for f in train_features], dtype=torch.long
        )
        all_target_ids = torch.tensor(
            [f.target_ids for f in train_features], dtype=torch.long
        )
        all_target_mask = torch.tensor(
            [f.target_mask for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_source_ids, all_source_mask, all_target_ids, all_target_mask
        )

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size // args.gradient_accumulation_steps,
        )

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
        )

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = (
            0,
            0,
            0,
            0,
            0,
            1e6,
        )
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                loss, _, _ = model(
                    source_ids=source_ids,
                    source_mask=source_mask,
                    target_ids=target_ids,
                    target_mask=target_mask,
                )

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss = round(
                    tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4
                )
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval and (epoch + 1) % args.save_inverval == 0:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False

                # Calculate bleu
                if "dev_bleu" in dev_dataset:
                    eval_examples, eval_data = dev_dataset["dev_bleu"]
                else:
                    eval_examples = read_examples(
                        args.dev_filename + "." + args.source,
                        args.dev_filename + "." + args.target,
                    )
                    eval_examples = random.sample(
                        eval_examples, min(1000, len(eval_examples))
                    )
                    eval_features = convert_examples_to_features(
                        eval_examples, tokenizer, args, stage="test"
                    )
                    all_source_ids = torch.tensor(
                        [f.source_ids for f in eval_features], dtype=torch.long
                    )
                    all_source_mask = torch.tensor(
                        [f.source_mask for f in eval_features], dtype=torch.long
                    )
                    eval_data = TensorDataset(all_source_ids, all_source_mask)
                    dev_dataset["dev_bleu"] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(
                    eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
                )

                model.eval()
                p = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask = batch
                    with torch.no_grad():
                        preds = model(source_ids=source_ids, source_mask=source_mask)
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[: t.index(0)]
                            text = tokenizer.decode(
                                t, clean_up_tokenization_spaces=False
                            )
                            p.append(text)
                model.train()
                predictions = []
                pred_str = []
                label_str = []
                with open(os.path.join(args.output_dir, "dev.output"), "w") as f, open(
                    os.path.join(args.output_dir, "dev.gold"), "w"
                ) as f1:
                    for ref, gold in zip(p, eval_examples):
                        ref = ref.strip().replace("< ", "<").replace(" >", ">")
                        ref = re.sub(
                            r' ?([!"#$%&\'(’)*+,-./:;=?@\\^_`{|}~]) ?', r"\1", ref
                        )
                        ref = ref.replace("attr_close>", "attr_close >").replace(
                            "_attr_open", "_ attr_open"
                        )
                        ref = ref.replace(" [ ", " [").replace(" ] ", "] ")
                        ref = ref.replace("_obd_", " _obd_ ").replace(
                            "_oba_", " _oba_ "
                        )

                        pred_str.append(ref.split())
                        label_str.append([gold.target.strip().split()])
                        predictions.append(str(gold.idx) + "\t" + ref)
                        f.write(str(gold.idx) + "\t" + ref + "\n")
                        f1.write(str(gold.idx) + "\t" + gold.target + "\n")

                bl_score = corpus_bleu(label_str, pred_str) * 100

                logger.info("  %s = %s " % ("BLEU", str(round(bl_score, 4))))
                logger.info("  " + "*" * 20)
                if bl_score > best_bleu:
                    logger.info("  Best bleu:%s", bl_score)
                    logger.info("  " + "*" * 20)
                    best_bleu = bl_score
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, "checkpoint-best-bleu")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_test:
        files = []
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(
                file + "." + args.source, file + "." + args.target
            )
            eval_features = convert_examples_to_features(
                eval_examples, tokenizer, args, stage="test"
            )
            all_source_ids = torch.tensor(
                [f.source_ids for f in eval_features], dtype=torch.long
            )
            all_source_mask = torch.tensor(
                [f.source_mask for f in eval_features], dtype=torch.long
            )
            eval_data = TensorDataset(all_source_ids, all_source_mask)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
            )

            model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    preds = model(source_ids=source_ids, source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[: t.index(0)]
                        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            predictions = []
            pred_str = []
            label_str = []
            with open(
                os.path.join(args.output_dir, "test_{}.output".format(str(idx))), "w"
            ) as f, open(
                os.path.join(args.output_dir, "test_{}.gold".format(str(idx))), "w"
            ) as f1:
                for ref, gold in zip(p, eval_examples):
                    ref = ref.strip().replace("< ", "<").replace(" >", ">")
                    ref = re.sub(r' ?([!"#$%&\'(’)*+,-./:;=?@\\^_`{|}~]) ?', r"\1", ref)
                    ref = ref.replace("attr_close>", "attr_close >").replace(
                        "_attr_open", "_ attr_open"
                    )
                    ref = ref.replace(" [ ", " [").replace(" ] ", "] ")
                    ref = ref.replace("_obd_", " _obd_ ").replace("_oba_", " _oba_ ")

                    pred_str.append(ref.split())
                    label_str.append([gold.target.strip().split()])
                    predictions.append(str(gold.idx) + "\t" + ref)
                    f.write(str(gold.idx) + "\t" + ref + "\n")
                    f1.write(str(gold.idx) + "\t" + gold.target + "\n")

            bl_score = corpus_bleu(label_str, pred_str) * 100
            logger.info("  %s = %s " % ("BLEU", str(round(bl_score, 4))))
            logger.info("  " + "*" * 20)