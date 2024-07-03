import Floater from './Floater';

function SectionTech() {
	const floaters = [
		[
			"Model Architecture",
			"FitGPT uses a reliable VGGNet-inspired architecture, a series of convolutional blocks followed by classifier layers. A pretrained ResNet FPN locates the person within the input image, telling the model where to look.",
			"InfoIcon"
		],
		[
			"Training Process",
			"FitGPT has been trained on a diverse array of fashion brand catalogs, enabling it to infer stylish colors from subjects' body shape and features. We used the AdamW optimizer with stepping learning rate decay.",
			"TrainTrackIcon"
		],
		[
			"Inference Speed",
			"Only a few MB in file size, our demo model uses the ONNX runtime for optimal performance. The ResNet FPN, rather than our model itself, contributes the most to inference time.",
			"TypeIcon"
		]
	]
	
	return (
		<div className="space-y-6">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Technical Details</h2>
			<p className="text-gray-400">
				We reduce the task of outfit recommendations to a set of image classification problems, simplifying the AI architecture involved.
				<br /> <br />
				Note: despite our product's name, we do not use a generative pre-trained transformer.
			</p>
			<div className="grid grid-cols-1 gap-6 sm:grid-cols-1 lg:grid-cols-3">
				{floaters.map(([titleText, description, iconSrc]) => (
					<Floater
						key={titleText}
						titleText={titleText}
						description={description}
						iconSrc={iconSrc}
						boxBg={true}
					/>
				))}
			</div>
		</div>
	)
}

export default SectionTech;
